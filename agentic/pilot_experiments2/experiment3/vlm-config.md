# 实验 3 的 VLM 配置（冻结稿，待用户确认冻结）

状态：**冻结稿，2026-11-28，提交 bbc45bc（代码、提示词、阈值、预算），待用户确认冻结**。自检结果见 README“接 VLM：冻结稿”一节。用户确认之前不算冻结；确认时在这里写日期和提交号，之后改动就是新配置、要重跑。调试只在开发布局（`exp3_harness.json` 的 `dev_seeds`）上做，不在 60 个官方评测回合上做。

下面的提示词块（` ```prompt ` 那一段）就是代码读的那一份（`kimodo_vla/agent/commander.py` 的 `load_prompt`），不另存副本。

## 已定的（用户 2026-11-28：“都按建议”；模型和密钥除外）

1. 底层只用 HEFT（`--tracker heft`）+ 运动学规划器（HITCH，`planner.name: hitch`）。SONIC 这一轮不跑：脚本指令在 SONIC + 运动学规划器上一次也没走到过抓取站位（README“HEFT 对比”：0/4 到站位）。
2. 任务先只做 P&PBox；SitSofa 以后再说（运动学规划器没有“坐到身后的座面上”这种姿态，见下面的指令表）。
3. 模型：先用本地 Qwen3-VL-4B-Instruct 把流程跑通。闭源模型：用户 2026-11-28 定“沿用之前的模型”，即第一批指点实验用的那两个（确切版本见下表）；密钥由用户自己填进开发机的 `agentic/env.local.sh`。闭源的 60 个正式回合要等用户看过用量估计、点头后再跑。
4. VLM 看到的数值和脚本指令一样（同一份观测），包括货架的真实（转过的）位姿和目标层上方到上一层隔板底面的净空；相机只给头部前视一张图。
5. 提示词里“两手夹住物体”那一句保留。
6. 成功以“松手 5 s 后箱子还在层板上、两手已离开”为准；Arena 自带的判定并列记录（判定程序的 `success` 和 `arena_success`）。
7. 阈值：`exp3_thresholds.json` 的草案值原样定死（2026-11-28）；只把第 4 环的“稳定放置”改成第 6 条的 5 s 标准。
8. 每回合上限：40 条指令、60 次调用、7000 步（140 s 仿真时间，`--steps 7000`）。
9. 判定的三条规则按现在的实现不动，**按草案，待用户确认**（见文末“按草案、待用户确认的”）。

## 模型和档位

| 项 | 开源 | 闭源前沿 |
|---|---|---|
| 模型 | Qwen3-VL-4B-Instruct，本地 bf16，权重目录见 `--vlm-model` | `gpt-6.2-astra`（OpenAI，指令来源 `gpt`）和 `gemini-4-argon`（Google，指令来源 `gemini`），和第一批指点实验相同（`pilot_experiments/experiment3_pointing/pointing-backends.json`）；连通调用里接口回报的版本号也是这两个 |
| 推理档位 | 贪心解码（`do_sample=False`），最多 400 个输出 token | 和指点实验一样不另设（`api.request` 为空 = 厂商默认的推理档位、温度和输出上限）。三个模型用同一份提示词 |
| 一次调用看到什么 | 一张头部前视相机 RGB 图（`front_camera`，仿真暂停时拍，和下面的数值同一时刻）+ 提示词 | 同左 |
| 每次输出 | 一条身体指令（JSON） | 同左 |
| 回复不合法 | 拒收，把原因告诉它再问；连续 3 次不合法结束回合（`budgets.max_refused_in_row`） | 同左 |
| 连不上 | — | 没回应或接口临时出错（408、429、5xx）就重发，最多 4 次（等 5、15、30、60 s），单独计数（`transport_retries`），不算模型的错；4 次都不行，回合以 `vlm_unreachable` 结束，按基础设施故障原样重跑。日志不记请求头和密钥，错误内容里形似密钥的字符串会被去掉 |
| 规划器拒收 | 指令合法但规划器做不了（例如手的目标超出从站的地方能够到的范围）：身体不动，规划器的原话作为错误告诉它，再问；不算进“连续不合法” | 同左 |
| 预算 | 每回合最多 40 条指令、60 次调用、7000 步（140 s） | 同左 |

## 它能看到的数值（全部是仿真真值，和脚本指令看到的是同一份）

- 机器人：骨盆位置（世界坐标）、朝向角、躯干前倾角、两个手腕的位置和掌心 / 手指方向、两只手开还是合。
- 物体（世界坐标，x、y 在地面，z 朝上，单位米，朝向角从 +x 逆时针）：
  - 箱子：中心、朝向（角度和四元数）、尺寸（0.21 m 立方）；
  - 桌子：轴对齐包围盒、台面高度；
  - 货架：轴对齐包围盒，加上 `oriented`，即货架框架的真实位姿（中心、朝向角、尺寸；Arena 每个布局把货架转最多 ±15°）。附说明“boxes go in through its long sides”，这是货架的几何事实：长边敞开、端部有立柱；
  - 货架目标层：要放的区域，Arena 判定就用它，说明是“中心落在这个区域上方、底面落在这个高度”。另有 `oriented`（隔板的真实位姿）和 `clear_height_m`（目标层面到上一层隔板底面的净空）。
- 上一条指令的结果：完成与否、用时、目标、实测与目标的偏差；被规划器拒收时是拒收的原话。还有最近 6 条指令的类型和剩余预算。

货架的 `oriented` 和 `clear_height_m` 从 fdb865c（2026-11-27）起就在观测里（为脚本指令加的）。VLM 用同一份观测，所以这次没有改代码，只在自检时核对它们确实进了提示词（见 README）。

## 它能发的指令（按运动学规划器能做的写）

`kimodo_vla/agent/commands.py` 的指令格式不变，取值范围在 `exp3_harness.json` 的 `limits`。经运动学规划器（`kimodo_vla/agent/kinematic.py`）执行时：

| 指令 | 规划器怎么做 | 和指令格式允许的范围的差别 |
|---|---|---|
| `walk` | 一条最多 1.9 m（VLM 指令来源的 `limits.walk_max_m`；HITCH 一次请求的走路上限是 2.0 m 脚步路径，直走 2.0 m 被拒过）。HITCH 走路规划；走完按真实骨盆位置再补走，最多 2 次，误差到 8 cm、6° 以内为止（`walk_tolerance_m`、`turn_tolerance_deg`）。两只手都张开时，先把手放到身体两侧再走；有一只手合着，手臂保持原样 | 指令格式共用的上限是 2 m，VLM 的收到 1.9 m |
| `turn` | 原地转，转角取整到 1° | 无 |
| `pelvis_height` | 蹲下 / 站起，HITCH 的高度范围 0.30–0.79 m | VLM 的指令范围收到 0.40–0.79（共用的是 0.40–0.80）；**没有 `behind_feet_m`**（坐到身后的姿态），带了这一项会被规划器拒收 |
| `torso_pitch` | 躯干前倾，按 5° 取整 | 指令允许 0–45° 任意值 |
| `hand_pose` | 手腕目标位姿，变成 HITCH 的掌心路径点。HITCH 嫌给的时长太短时，按它需要的时长再规划一次；够不到时，自己前倾 10°、20° 再试 | 无 |
| `hand` | 开 / 合，一只或两只 | 无 |
| `done` | 结束回合 | 无 |

被规划器拒收（够不到、关节超限、HITCH 的检查没过）的指令不执行，身体保持不动，拒收原话作为错误回给模型再问（`kimodo_vla/agent/loop.py`）。原话包括 HITCH 给的明细（哪一项超限、要了多少、上限多少），没有明细时只有那句话。

指令格式本身的检查（`commands.py`，不合法就拒收、告诉原因再问）还有两条，现在也写进了提示词：手腕目标离骨盆不超过 1.2 m（`workspace_radius_m`），掌心方向和手指方向至少差 45°（`min_palm_finger_angle_deg`）。VLM 的指令来源在 `exp3_harness.json` 里有自己的 `limits`（走路 1.9 m、骨盆 0.40–0.79），脚本指令仍用共用的那份，参考策略不受影响。

## 和上一版草案（KimodoPlanner）比，提示词改了什么

只改了和规划器、指令检查有关的事实，没有加脚本规则里的解题步骤或调出来的数（站位距离、撤手 6 cm、先放到层板上再张手这类都没有写进去）。逐条：

1. **删去 `pelvis_height` 带 `behind_feet_m` 的那一行**（“坐到身后的座面上”）。理由：运动学规划器没有这个姿态，发了一定被拒收；这一轮只做 P&PBox，用不到坐下。
2. **`pelvis_height` 的范围写成 0.40–0.79**（原来 0.40–0.80）。理由：HITCH 的上限是 0.79 m。
3. **`torso_pitch` 注明“按 5° 取整”**。理由：HITCH 的躯干姿态只接受 5° 的整数倍，规划器会取整。
4. **走路的精度改成“走完后按 8 cm、6° 以内补走”**（原来写“走完离目标 5–10 cm、转身差几度”，那是 Kimodo 走路的经验数）。理由：运动学规划器的走路是闭环补走，这两个数是它的设置值，不是按结果调出来的。
5. **加一句：两只手都张开时，走路前先把手放到身体两侧**。理由：这是规划器自己的动作（模型会看到手腕位置变了），不说明会让它误以为手被别的东西碰了。
6. **加一句：规划器做不了的指令不执行，原因会告诉它**。理由：这是接口行为，模型需要知道被拒的指令没有执行。
7. **走路的上限写成 1.9 m**（原来 2 m），自检后改。理由：自检第一轮（v1heft）三个布局的第一条指令都是直走 2.0 m，都被 HITCH 拒收（一次请求的走路超出范围）；HITCH 的上限是 2.0 m 脚步路径，脚步路径比直线略长。VLM 的指令检查同时收到 1.9 m，确切边界没测。
8. **`hand_pose` 那一行加上“离骨盆不超过 1.2 m、掌心和手指至少差 45°”**，自检后改。理由：这是指令检查本来就有的两条规则，提示词里没写；自检第一轮里 2 个回合各连续 3 次发了掌心和手指方向相同的伸手指令，被拒收到回合结束。只把规则写出来，没有写怎么定掌心方向。
9. **没改的**：身体描述、“两手夹住物体”那一句（用户定保留）、坐标约定、其余指令格式、输出要求。

## 提示词全文

`{observation}` 换成当时的观测 JSON，`{task}` 换成任务说明，`{error}` 换成上一条被拒的原因（回复不合法，或规划器拒收；没有就是空）。

```prompt
You control a Unitree G1 humanoid robot in a physics simulator, one body command at a time.
After each command is carried out you will get a new camera image and a new observation, and you choose the next command.

THE ROBOT
- About 1.3 m tall. Standing, its pelvis is about 0.78 m above the floor; the shoulders are about 0.3 m above the pelvis and 0.2 m to either side of its centre line.
- Each arm reaches about 0.55 m from its shoulder. The hands can only open or close (no finger control). To hold an object between two hands, put the palms against opposite faces of it and close both hands.
- It walks with a learned balance controller. After a walk or turn, the remaining error is walked off again until the robot is within about 8 cm and 6 degrees of the commanded spot. It cannot jump or climb. With both hands open, it first brings the hands to its sides before walking; with a hand closed, it walks with its arms held as they were.
- Everything you command is executed whole before you are asked again; the simulator is paused while you think.
- A command the motion planner cannot carry out (for example a hand target beyond the arm's reach from where the robot stands) is not executed; you are told the planner's reason and choose again.

FRAMES AND UNITS
- World frame: x and y on the floor, z up, metres, floor at z = 0. Angles in degrees, counter-clockwise seen from above, 0 = the world +x direction.
- "walk" and "turn" are relative to the robot as it stands: forward_m along its facing direction, left_m to its left, left_deg counter-clockwise.
- Every other position or direction (hand targets, palm and finger directions) is in the world frame.
- The robot's facing direction is heading_deg in the observation; "palm" is the direction a palm faces, "fingers" the direction the straight fingers point.

COMMANDS (reply with exactly one, as one JSON object, and nothing else)
{"type": "walk", "forward_m": <-1.9..1.9>, "left_m": <-1.9..1.9>}    total distance 0.02 to 1.9 m
{"type": "turn", "left_deg": <-180..180>}                            at least 2 degrees
{"type": "pelvis_height", "height_m": <0.40..0.79>}                  crouch or stand, feet stay, body balanced over the feet
{"type": "torso_pitch", "pitch_deg": <0..45>}                        lean the upper body forward (0 = upright), in steps of 5 degrees
{"type": "hand_pose", "hands": {"left": {"position": [x, y, z], "palm": [x, y, z], "fingers": [x, y, z]}, "right": {...}}, "duration_s": <0.5..4>}
                                                                     move one or both wrists; a hand left out keeps its pose; palm and fingers may be left out to keep the orientation
                                                                     each position at most 1.2 m from the pelvis; palm and fingers at least 45 degrees apart
{"type": "hand", "left": "open" | "close", "right": "open" | "close"}   either or both
{"type": "done"}                                                     when the task is complete, or cannot be completed

TASK
{task}

OBSERVATION (ground truth from the simulator)
{observation}

{error}Reply with one JSON command.
```

## 判定（改完的）

- **成功**（P&PBox，2026-11-28 定）：某次张手指令（在第一次合手之后）发出后 5 s，Arena 的在架判定在前后 10 个记录步上都为真，且两只手对箱子的力都小于 1 N。读法和 `exp3_release_check.py` 相同。代码是 `kimodo_vla/agent/judge.py` 的 `released_and_kept`，阈值是 `place_hold_s`、`place_window_steps`、`place_hands_off_n`。Arena 自己的判定记为 `arena_success`。
- 第 4 环“放稳”（放置阶段）用同一条读法，替换原来的“张手后 1 s 内 Arena 判过一次在架”（`place_settle_s`，已删）。
- 其余阈值原样定死，标“fixed 2026-11-28 (draft value)”。

## 按草案、待用户确认的

- **判定的三条规则**（按现在的实现，没有改）：
  - ① 已经抬起箱子的那次尝试，它的第 1–3 环检查只记录、不当失败（标 superseded）；
  - ② 一次都没尝试的回合，途中有碰撞、摔倒或过度倾斜就归第 1 环，否则“未归类”；
  - ③ 多次尝试按最早那次失败归环（`attribution='earliest'`，另有 `last_attempt` 可选）。
- **第 5 环“抬起后丢箱”的一个问题**（改第 4 环、重判已有日志时发现的，没有改）：它看箱子相对两手中点的偏移变化，偏移是在世界坐标里算的。搬着箱子转身时，箱子没动也会算出偏移，超过 8 cm 就误判成丢箱。例子：第一次测量的 g2r08 在 27.5–30.8 s 转身，两只手一直各用约 20 N 夹着箱子、箱子高度不变，偏移却变了 11 cm。建议改成跟着身体朝向转的坐标，要不要改等用户定。
- **闭源模型**：用哪个、确切版本号、推理档位、API 预算（用户还没定，密钥没填）。
- **相机**：现在只给头部前视一张图（第 4 条已定）。要不要以后再加一张俯视的世界相机图，留作以后的选择。
