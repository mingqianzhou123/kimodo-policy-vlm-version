# 实验 3 的 VLM 配置（草案，待团队冻结）

状态：草案，2026-11-26。冻结前只在开发布局（`exp3_harness.json` 的 `dev_seeds`）上调试，不在 V 上调。冻结时在这里写日期和提交号，之后改动就是新配置、要重跑。

下面的提示词块（` ```prompt ` 那一段）就是代码读的那一份（`kimodo_vla/agent/commander.py` 的 `load_prompt`），不另存副本。

## 模型和档位

| 项 | 开源 | 闭源前沿 |
|---|---|---|
| 模型 | Qwen3-VL-4B-Instruct，本地 bf16，权重目录见 `--vlm-model` | **待定**：GPT 或 Gemini 的确切版本号（第一批指点实验用过 GPT-6.2 Astra、Gemini 4 Argon，版本号写在 `pilot_experiments/experiment3_pointing/pointing-backends.json`） |
| 推理档位 | 贪心解码（`do_sample=False`），最多 400 个输出 token | 待定：推理档位、温度、输出上限，写进 `exp3_harness.json` 的 `commanders.<名字>.api.request` |
| 一次调用看到什么 | 一张头部前视相机 RGB 图（`front_camera`，仿真暂停时拍，和下面的数值同一时刻）+ 提示词 | 同左 |
| 每次输出 | 一条身体指令（JSON） | 同左 |
| 回复不合法 | 拒收，告诉它原因再问；连续 3 次不合法结束回合（`budgets.max_refused_in_row`） | 同左 |
| 预算 | 每回合最多 40 条指令、60 次调用；Arena 步数上限 3000 步（60 s 仿真时间） | 同左 |

## 它能看到的数值（全部是仿真真值）

- 机器人：骨盆位置（世界坐标）、朝向角、躯干前倾角、两个手腕的位置和掌心 / 手指方向、两只手开还是合。
- 物体（世界坐标，x、y 在地面，z 朝上，单位米，朝向角从 +x 逆时针）：
  - P&PBox：箱子（中心、朝向、尺寸 0.21 m 立方）、桌子（轴对齐包围盒和台面高度）、货架（包围盒）、货架目标层（要放的那一层：中心落在这个区域内、底面落在这个高度，Arena 的判定就是这个）。
  - SitSofa：座面（轴对齐包围盒，Arena 判定坐下用的那个盒子）、两个纸箱障碍物（中心和尺寸）。
- 上一条指令的结果：完成与否、用时、目标、实测与目标的偏差；最近 6 条指令的类型。

## 它能发的指令

`kimodo_vla/agent/commands.py` 的七种：走路位移、转身、骨盆高度（可选“骨盆落到脚后方多远”，用来坐下）、躯干前倾、手部目标位姿（位置 + 掌心朝向 + 手指方向，一只或两只手）、开合手、结束。每条的取值范围在 `exp3_harness.json` 的 `limits`。

## 指令经哪个规划器变成 40 维参考

`kimodo_vla/agent/planner.py` 的 `KimodoPlanner`：走路和转身用实验二验证过的官方方式 Kimodo 走路；骨盆高度和躯干前倾用新写的姿态求解（双脚不动）加一段 Kimodo 生成；手部位姿用第一批抓箱的站立 IK 投影后端；开合手直接改 40 维里的两位。规划器是一个接口，实验 1 的结论如果要求换运动学规划器，只换这一层。

## 提示词全文

`{observation}` 换成当时的观测 JSON，`{error}` 换成上一条被拒的原因（没有就是空）。

```prompt
You control a Unitree G1 humanoid robot in a physics simulator, one body command at a time.
After each command is carried out you will get a new camera image and a new observation, and you choose the next command.

THE ROBOT
- About 1.3 m tall. Standing, its pelvis is about 0.78 m above the floor; the shoulders are about 0.3 m above the pelvis and 0.2 m to either side of its centre line.
- Each arm reaches about 0.55 m from its shoulder. The hands can only open or close (no finger control). To hold an object between two hands, put the palms against opposite faces of it and close both hands.
- It walks with a learned balance controller: walks end within about 5-10 cm of the commanded spot, turns within a few degrees. It cannot jump or climb. Carrying something, it walks with its arms held as they were.
- Everything you command is executed whole before you are asked again; the simulator is paused while you think.

FRAMES AND UNITS
- World frame: x and y on the floor, z up, metres, floor at z = 0. Angles in degrees, counter-clockwise seen from above, 0 = the world +x direction.
- "walk" and "turn" are relative to the robot as it stands: forward_m along its facing direction, left_m to its left, left_deg counter-clockwise.
- Every other position or direction (hand targets, palm and finger directions) is in the world frame.
- The robot's facing direction is heading_deg in the observation; "palm" is the direction a palm faces, "fingers" the direction the straight fingers point.

COMMANDS (reply with exactly one, as one JSON object, and nothing else)
{"type": "walk", "forward_m": <-2..2>, "left_m": <-2..2>}            total distance 0.02 to 2 m
{"type": "turn", "left_deg": <-180..180>}                            at least 2 degrees
{"type": "pelvis_height", "height_m": <0.40..0.80>}                  crouch or stand, feet stay, body balanced over the feet
{"type": "pelvis_height", "height_m": <0.40..0.80>, "behind_feet_m": <0..0.30>}   the pelvis ends this far behind the feet: this is how to sit down on a seat behind you
{"type": "torso_pitch", "pitch_deg": <0..45>}                        lean the upper body forward (0 = upright)
{"type": "hand_pose", "hands": {"left": {"position": [x, y, z], "palm": [x, y, z], "fingers": [x, y, z]}, "right": {...}}, "duration_s": <0.5..4>}
                                                                     move one or both wrists; a hand left out keeps its pose; palm and fingers may be left out to keep the orientation
{"type": "hand", "left": "open" | "close", "right": "open" | "close"}   either or both
{"type": "done"}                                                     when the task is complete, or cannot be completed

TASK
{task}

OBSERVATION (ground truth from the simulator)
{observation}

{error}Reply with one JSON command.
```

## 冻结前要定的

- 闭源模型用哪个、确切版本号、推理档位和 API 预算（一次连通测试之外不花闭源预算，等用户同意）。
- 提示词里的身体描述和“两手夹住物体”这一句提示是否保留：它给了抓取方式的常识，可能让第 2 类失败变少。这是选择，不是事实。
- 预算（40 条指令、60 次调用、60 s 仿真时间）和不合法回复的处理（拒收重问，连续 3 次结束）。
- 相机：现在只给头部前视一张图。要不要再给一张俯视的世界相机图。
