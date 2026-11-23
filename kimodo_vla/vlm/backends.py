"""Three interchangeable image+text backends: a resident local Qwen, and OpenAI / Gemini over HTTPS.

Every backend has ``generate(image, prompt, max_tokens)`` and returns one record with the reply text,
the latency of that call, token counts and whatever is needed to reproduce it later. API keys are read
from the environment and never stored.
"""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class QwenBackend:
    """Loads the weights once and keeps them on the GPU, so a call's latency is inference only."""
    name = 'qwen'

    def __init__(self, model_path):
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval()
        manifest = model_path/'download-manifest.json'
        self.info = {'model': model_path.name, 'model_version': json.loads(manifest.read_text()) if manifest.is_file() else None,
                     'load_s': time.perf_counter()-start, 'transformers_version': transformers.__version__,
                     'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                     'weights_memory_mib': torch.cuda.memory_allocated()/2**20}
        self.generate(Image.new('RGB', (640, 480)), 'Describe the image in one word.', 8)  # first-call set-up is not a latency sample
        torch.cuda.reset_peak_memory_stats()

    def generate(self, image, prompt, max_tokens):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        text = self.processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        torch.cuda.synchronize()
        return {'text': text, 'latency_s': time.perf_counter()-start, 'prompt': prompt, 'max_tokens': max_tokens,
                'image_sha256': hashlib.sha256(png_bytes(image)).hexdigest(),
                'input_tokens': int(inputs['input_ids'].shape[1]), 'output_tokens': int(output.shape[1]-inputs['input_ids'].shape[1])}

    def memory(self):
        return {'peak_allocated_mib': self.torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': self.torch.cuda.max_memory_reserved()/2**20}


def post_json(url, headers, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(), {**headers, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {'error_body': exc.read().decode(errors='replace')[:2000]}


class ApiBackend:
    """Shared part of the hosted backends. The exact model version comes from the config; there is no default."""
    key_variable = None

    def __init__(self, config, post=post_json):
        if not config.get('model'):
            raise SystemExit(f"Set the exact model version for '{self.name}' in agentic/configs/pointing-backends.json (field 'model')")
        self.key = os.environ.get(self.key_variable)
        if not self.key:
            raise SystemExit(f'{self.key_variable} is not set: put it in agentic/env.local.sh and source that file')
        self.config, self.post = config, post
        self.info = {'model': config['model'], 'base_url': config['base_url'], 'request': config.get('request', {}),
                     'price_per_million_tokens_usd': config.get('price_per_million_tokens_usd')}

    def cost(self, input_tokens, output_tokens):
        price = self.config.get('price_per_million_tokens_usd') or {}
        if None in (input_tokens, output_tokens, price.get('input'), price.get('output')):
            return None
        return (input_tokens*price['input']+output_tokens*price['output'])/1e6

    def generate(self, image, prompt, max_tokens):
        # max_tokens is the local model's budget. Hosted models take their limits from config['request'],
        # because a reasoning model spends an unknown share of any cap on hidden tokens.
        data = png_bytes(image)
        url, headers, body = self.request(prompt, base64.b64encode(data).decode())
        start = time.perf_counter()
        try:
            status, response = self.post(url, headers, body, self.config.get('timeout_s', 120))
        except OSError as exc:  # no reply at all; the elapsed time is still the cost of this call
            status, response = None, {'error_body': repr(exc)}
        latency = time.perf_counter()-start
        record = {'text': '', 'latency_s': latency, 'prompt': prompt, 'image_sha256': hashlib.sha256(data).hexdigest(),
                  'http_status': status, 'response': response, 'input_tokens': None, 'output_tokens': None,
                  'model_version': None, 'request': self.config.get('request', {})}
        if status == 200:
            record.update(self.read(response))
        record['cost_usd'] = self.cost(record['input_tokens'], record['output_tokens'])
        return record


class OpenAIBackend(ApiBackend):
    name, key_variable = 'openai', 'OPENAI_API_KEY'

    def request(self, prompt, image_b64):
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+image_b64}}, {'type': 'text', 'text': prompt}]
        return (self.config['base_url'].rstrip('/')+'/chat/completions', {'Authorization': 'Bearer '+self.key},
                {'model': self.config['model'], 'messages': [{'role': 'user', 'content': content}], **self.config.get('request', {})})

    def read(self, response):
        usage = response.get('usage') or {}
        return {'text': response['choices'][0]['message'].get('content') or '', 'model_version': response.get('model'),
                'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens')}


class GeminiBackend(ApiBackend):
    name, key_variable = 'gemini', 'GEMINI_API_KEY'

    def request(self, prompt, image_b64):
        parts = [{'inline_data': {'mime_type': 'image/png', 'data': image_b64}}, {'text': prompt}]
        return (f"{self.config['base_url'].rstrip('/')}/models/{self.config['model']}:generateContent", {'x-goog-api-key': self.key},
                {'contents': [{'role': 'user', 'parts': parts}], **self.config.get('request', {})})

    def read(self, response):
        usage = response.get('usageMetadata') or {}
        parts = ((response.get('candidates') or [{}])[0].get('content') or {}).get('parts') or []
        billed = [usage.get(k) for k in ('candidatesTokenCount', 'thoughtsTokenCount') if usage.get(k) is not None]
        return {'text': ''.join(p.get('text', '') for p in parts if not p.get('thought')), 'model_version': response.get('modelVersion'),
                'input_tokens': usage.get('promptTokenCount'), 'output_tokens': sum(billed) if billed else None}


API_BACKENDS = {'openai': OpenAIBackend, 'gemini': GeminiBackend}
