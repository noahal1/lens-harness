# lens-harness

一个**可读的、LangGraph 风格的编排内核**，配一份能被回放、diff、时间旅行的 trace。

它不是「又一个 agent 框架」。它想回答的是另一个问题：

> 一个 agent 框架在两次 `await` 之间，到底替你做了什么决定？

`agent → tools → agent → END` 这条链路每个框架都有，但调度过程通常藏在日志、回调或者干脆藏起来。lens-harness 把这层摊开：每一步调度都写成一个 `Event`，写进一份 JSONL，然后有一个**只读 trace、不 import 内核**的渲染器把它画成人能读的样子。

```
┌─ kernel: lens_harness/ ──────────────┐      ┌─ lens: 只读渲染 ─┐
│ events  state  graph  runtime        │      │ render.py        │
│ checkpoint  llm                      │ ───▶ │ (终端渲染器)      │
│        └── Emitter → EventSink ──────┘trace │                  │
└──────────────────────────────────────┘      └──────────────────┘
```

三条设计原则（详见 [DESIGN.md](DESIGN.md)）：

1. **Trace 是唯一事实来源** —— 状态在哪、谁写了什么、路由为什么这样选、checkpoint 何时落盘，全部只通过事件表达，没有旁路。
2. **UI 是 trace 的纯函数** —— 渲染器只吃 `list[Event]`，从不 import `graph` / `runtime`。这是验收标准，不是洁癖。
3. **回放不需要重跑 LLM** —— `node_token` 事件把流式输出逐块记进 trace，配合确定性 `MockLLM`，一份 trace 可以离线重放。

---

## 环境要求

| 项 | 值 |
| --- | --- |
| 解释器 | **`py -3.13`**（实测 Python 3.13.14） |
| 依赖 | **零**。纯标准库，不联网也能跑 |
| 虚拟环境 | **不需要，也不要建**（本机 `ensurepip` 失败） |
| 操作系统 | Windows（macOS / Linux 同样可用，把 `py -3.13` 换成 `python3.13`） |

> 本文所有命令一律写 `py -3.13`，这是本项目钉住的版本。补充一个实测到的细节：本机 **裸 `py`（不带版本）当前解析到 Python 3.14.0**，用它跑 `tests.test_kernel` 也是 25/25 通过（`tests.test_flows` 8/8）。也就是说 3.14 大概也能跑，但项目是按 3.13 验的，请继续用 `py -3.13`。

### ⚠️ `python` / `python3` 是坏的，必须用 `py`

```console
$ py -3.13 -V
Python 3.13.14                          # 退出码 0

$ python -V
                                        # 没有任何输出，退出码 9009

$ python3 -V
                                        # 没有任何输出，退出码 9009
```

（上面三行的「退出码」都是**在同一个 shell 里紧跟一条 `echo $?`** 得到的，这是唯一可靠的测法：`python -V` 本身什么都不打印，光看屏幕分不出「成功但无输出」和「根本没启动」。）

> 一个反直觉的细节，**别被它骗了**：从 Python 里 `subprocess.run(["python","-V"])` 实测会**成功**（返回码 0、打印 `Python 3.13.14`），但同一个调用里 `python3` 仍然返回 9009；`subprocess.run("python -V", shell=True)` 两个又都返回 9009。同一台机器、同一条命令，取决于从哪儿发起，结果不一样。
> 所以别用「`subprocess` 跑一下看返回码」来探这个环境——结论会随调用方式漂移。**要确认就老老实实在 shell 里敲 `py -3.13`。**

`which python` 指向 `C:/Users/noaha/AppData/Local/Microsoft/WindowsApps/python.exe` —— 这是 **Windows Store 的「应用执行别名」占位程序（stub）**，不是真的解释器：它不启动任何 Python 进程，直接返回退出码 `9009`。`python3` 同理。

所以本文所有命令**一律用 `py -3.13`**，不要写 `python`。

`py` 是独立的 Python Launcher（`C:/Users/noaha/AppData/Local/Programs/Python/Launcher/py.exe`），它认得多个已注册版本：

```console
$ py -0
 -V:3.14 *        Python 3.14 (64-bit)
 -V:3.13          Python 3.13 (Store)
 -V:Astral/CPython3.12.13 CPython 3.12.13 (64-bit)
```

`*` 是裸 `py` 的默认版本（3.14）。**永远显式写 `-3.13`**，不要依赖默认值。

（注意 `-V:3.13` 那一行写着 `(Store)`——3.13 本身也是 Store 发行版，但通过 `py -3.13` 启动是正常的，别被这个词吓到。坏的只是 `WindowsApps/` 下那几个 stub。）

### 关于中文和 emoji 会不会崩

Windows 控制台默认代码页是 GBK，理论上打印 `▶ ▸ ⟳ ⑂ ◆ ■` 这些字符会 `UnicodeEncodeError`。**cli 内部已经强制把 stdout/stderr 重新配置成 UTF-8**（`errors="replace"` 兜底），所以正常打印不会崩。

实测验证过这条：把输出环境变量强制成 GBK 再跑，输出与 UTF-8 环境下**逐字节相同**（80 行、stderr 空、退出码 0）。下面两条在 MSYS bash 里跑（`/tmp` 是 bash 的临时目录）：

```console
$ PYTHONIOENCODING=gbk py -3.13 -m lens_harness show --no-color traces/r_c22281d6.jsonl > /tmp/gbk.txt
$ py -3.13 -m lens_harness show --no-color traces/r_c22281d6.jsonl > /tmp/norm.txt
$ cmp /tmp/gbk.txt /tmp/norm.txt && echo "BYTE-IDENTICAL to normal run"
BYTE-IDENTICAL to normal run
```

> 这只覆盖「cli 自己打印」这条路。**你自己写的脚本**里 `print('▶')` 在 GBK 控制台下照样会崩——那时在脚本开头加一句 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，和 cli 里 `_force_utf8_stdout` 的做法一样。

---

## 快速开始

在项目根目录执行，四条命令跑完你就理解了整个项目。

### 1. 跑测试（确认环境没问题）

```bash
py -3.13 -m tests.test_kernel
py -3.13 -m tests.test_flows
```

实测输出：

```
...
25/25 passed
```

```
...
8/8 passed
```

`test_kernel.py` 是内核自测（25 项），`test_flows.py` 是流程自测（8 项：interrupt/resume + LLM adapter），`test_serve.py` 是传输层自测（31 项：SSE 帧、断点续传、流必须真的结束、页面纯度）。都是纯标准库断言，**不需要 pytest**。

### 2. 跑一遍示例，生成一份 trace

```bash
py -3.13 -m lens_harness run examples.agent:build --input '{"question":"What is 17*23?"}'
```

这条命令**不用再单独 `show`** —— `cmd_run` 跑完示例后会立刻 `render_events` 把整份 trace 打到屏幕上。输出开头（渲染之前）会先有一行文件位置提示：

```
[trace written to traces\r_c22281d6.jsonl]
```

注意路径里用的是反斜杠（Windows 的 `os.path.join` 产物）。文件实际就在 `traces/r_c22281d6.jsonl`。

### 3. 渲染这份 trace

```bash
py -3.13 -m lens_harness show traces/r_c22281d6.jsonl
```

> `<run_id>` 每次运行都不一样，不要照抄上面的 `r_c22281d6`。取最新的一份见下面「怎么找到我刚刚跑的那份 trace」。

### 4. 看它的 checkpoint DAG

```bash
py -3.13 -m lens_harness dag traces/r_c22281d6.jsonl
```

```
checkpoint DAG
○ ss0 demo_0000_51d41f → tools
└─ ○ ss1 demo_0001_4e0d8d → agent
  └─ ○ ss2 demo_0002_caa18f → tools
    └─ ○ ss3 demo_0003_06131d → agent
      └─ ● ss4 demo_0004_425dc5 → end
```

`○` running，`●` 终态。`→` 后面是这一步之后要去哪。

---

## 命令速查

八个（`lens_harness/cli.py` 里注册的）：

| 命令 | 干什么 |
| --- | --- |
| `show <trace>` | 渲染整份 trace |
| `dag <trace>` | 打印 checkpoint DAG |
| `summary <trace>` | 按 kind 数事件 |
| `checkpoints <trace>` | 列出 checkpoint 及状态（已按 id 去重） |
| `state <trace> [ckpt_id]` / `state <trace> --at N` | 某个 checkpoint 的状态 |
| `watch <trace>` | 跟随一段正在写入的 trace |
| `diff <trace-a> <trace-b>` | 归一化后对比两份 trace（有差异退出码 1） |
| `serve <trace> [--host H] [--port P] [--follow]` | 在浏览器里看同一份 trace |
| `serve <trace-a> <trace-b>` | 浏览器版对比页（`/compare`） |
| `run <mod:attr> [--input JSON] [--trace-dir DIR] [--llm SPEC]` | 跑示例，写 trace 并渲染 |

下面的命令块里 `<trace>` 都换成你自己的那份 jsonl 路径（本文样例用 `traces/r_c22281d6.jsonl`）。

`show` 的四个开关：

```bash
# 注意：--no-tokens / --no-state / --no-diff 只属于 show，不是全局参数
py -3.13 -m lens_harness show --no-color --no-tokens traces/r_c22281d6.jsonl   # 隐藏 ↳ tokens 行
py -3.13 -m lens_harness show --no-color --no-state  traces/r_c22281d6.jsonl   # 隐藏结尾 final state
py -3.13 -m lens_harness show --no-color --no-diff   traces/r_c22281d6.jsonl   # 隐藏 ⟳ 下的 before → after
```

拿本文样例 trace 实测这三个开关（基准是完整渲染的 80 行，退出码都是 0）：

| 开关 | 输出行数 | 少掉的行 |
| --- | --- | --- |
| （不加） | 80 | — |
| `--no-tokens` | 77 | 3 条 `↳ tokens` |
| `--no-state` | 57 | 23 行 `final state:` 段 |
| `--no-diff` | 68 | 12 行 `⟳` 下的 before → after |

`--no-color` 是唯一**全局**的，放在子命令前后都行（两处都注册了）：

```bash
py -3.13 -m lens_harness --no-color show traces/r_c22281d6.jsonl   # ✅
py -3.13 -m lens_harness show --no-color traces/r_c22281d6.jsonl   # ✅ 等价
```

实测两条输出 `cmp` 下来逐字节相同。

`state` 的两种指定方式：

```bash
py -3.13 -m lens_harness state traces/r_c22281d6.jsonl demo_0002_caa18f   # 按 checkpoint id
py -3.13 -m lens_harness state traces/r_c22281d6.jsonl --at 2             # 按 superstep 号
```

这两条指向同一个 checkpoint，实测输出也一样。

`--at N` 是「产出 superstep N 的那个 checkpoint」的语法糖——你是从上往下读 trace 时发现某一步不对，那就说步号。

---

### `--help` 长什么样

忘了子命令叫什么就跑这个，它同时也是参数顺序的权威来源：

```console
$ py -3.13 -m lens_harness --help
usage: lens [-h] [--no-color]
            {show,dag,summary,checkpoints,state,watch,serve,run} ...

``python -m lens_harness`` — inspect a trace, or run an example.

Commands::

    python -m lens_harness show traces/<file>.jsonl        # render a trace
    python -m lens_harness dag traces/<file>.jsonl         # checkpoint DAG
    python -m lens_harness summary traces/<file>.jsonl     # event counts
    python -m lens_harness checkpoints traces/<file>.jsonl # list checkpoints
    python -m lens_harness state traces/<file>.jsonl [<ckpt_id>]
    python -m lens_harness state traces/<file>.jsonl --at <superstep>
    python -m lens_harness watch traces/<file>.jsonl       # follow a live run
    python -m lens_harness serve traces/<file>.jsonl       # the web view of a trace
    python -m lens_harness run examples.agent:build --input '{"question":"..."}'

``show``/``dag``/``checkpoints``/``watch``/``serve`` take only a trace. ``state``
takes the trace first and then optionally a checkpoint id, or ``--at N`` for the
state that superstep produced. ``--no-color`` works before or after the
subcommand. ``serve`` opens a browser page that is a pure function of the same
trace the terminal renderer reads; ``--follow`` keeps reading a run in progress.

positional arguments:
  {show,dag,summary,checkpoints,state,watch,serve,run}
    show                render a trace
    dag                 print the checkpoint DAG
    summary             event counts
    checkpoints         list checkpoints
    state               state at a checkpoint
    watch               follow a live trace
    serve               render a trace in the browser
    run                 run an example that returns events

options:
  -h, --help            show this help message and exit
  --no-color            disable ANSI colors
```

（`usage:` 下面那段里 `` ``python -m lens_harness`` `` 的双反引号是 docstring 的 RST 标记被原样打出来了，无伤大雅。）

## 目录结构

```
lens-harness/
├── lens_harness/          # 内核 + 渲染器
│   ├── events.py          # 事件契约：Event、EVENT_SCHEMA、12 种事件、sink、read/follow
│   ├── state.py           # channel、reducer（overwrite/add/add_unique/merge_dict）、patch
│   ├── graph.py           # StateGraph 声明与编译、NodeContext、START/END
│   ├── runtime.py         # ★ superstep 循环（整套东西的心脏）
│   ├── checkpoint.py      # Checkpoint / CheckpointStore，时间旅行靠它
│   ├── render.py          # ★ 纯函数渲染器：只吃 list[Event]
│   ├── compare.py         # ★ M3 trace 对比：归一化 + 对齐 + diff（只 import events）
│   ├── serve.py           # 把同一份 trace 用 SSE 推给浏览器（第 12 节）；双 trace 时加 /compare + /api/diff
│   ├── web/
│   │   ├── index.html     # ★ 浏览器渲染器：单文件、零依赖、零构建
│   │   └── compare.html   # ★ M3 对比页：吃 /api/diff 的 JSON，同样零依赖
│   ├── llm.py             # MockLLM / OpenAICompatibleLLM / RecordingLLM
│   ├── cli.py             # 上面那九个子命令
│   └── __init__.py        # 公开面（__version__ = "0.1.0"）
├── examples/
│   ├── agent.py           # tool-calling loop：agent ⇄ tools
│   └── approval.py        # human-in-the-loop：interrupt → 人类回复 → resume
├── tests/
│   ├── test_kernel.py     # 30 项，内核契约
│   ├── test_flows.py      # 8 项，interrupt/resume + LLM
│   ├── test_serve.py      # 31 项，SSE 传输 + 页面纯度
│   └── test_compare.py    # 8 项，M3 trace 对比契约
├── traces/                # 运行时产物
│   ├── r_XXXXXXXX.jsonl   # 一份 trace（run_id = "r_" + uuid4 hex[:8]）
│   └── checkpoints/       # <thread_id>_<superstep:04d>_<hex6>.json
├── DESIGN.md              # 设计文档（12 章 + 已知落差自查）
└── README.md              # 你正在读的
```

建议的源码阅读顺序（`__init__.py` 的 docstring 也是这么写的）：
`events.py` → `state.py` → `graph.py` → `runtime.py` → `checkpoint.py` → `render.py` → `serve.py` → `web/index.html`。

前六个看「机制怎么跑」，后两个看「同一份事件怎么变成眼睛能看到的东西」。

---

## 读一份真实 trace

下面是**真实运行输出**，不是手写的示意。命令：

```bash
py -3.13 -m lens_harness show --no-color traces/r_c22281d6.jsonl
```

（`--no-color` 只是为了让下面的文本里没有 ANSI 转义；不加时会带颜色，内容一样。）

### 完整输出（80 行）

```text
▶ run r_c22281d6  thread=demo
  graph: tool-loop-agent  entry: agent
  nodes: agent, tools
  channels: question[overwrite], messages[add], tools_used[add_unique], answer[overwrite], steps[overwrite]
  inputs: answer=, messages=[{"role": "user", "content": "What is 17…, question=What is 17*23?, steps=0, tools_used=[]

── superstep 0 ──────────────────────────────────────────────
  plan: agent
  ▸ agent  0.0 ms
    writes: messages=[{"role": "assistant", "content": "Let me compute that. TOOL…, steps=1
    ↳ tokens agent: 'Let me compute that. TOOL: calculator | 17*23'
  ⟳ state changed: messages, steps
    messages: [{"role": "user", "content": "What is 17*23?"}] → [{"role": "user", "content": "What is 17*23?"}, {"role": "as…
    steps: 0 → 1
  ⑂ route after agent via route_after_agent()
    candidates: tools | __end__
    returned:   'tools'
    selected:   tools
  ◆ checkpoint demo_0000_51d41f running next=tools

── superstep 1 ──────────────────────────────────────────────
  plan: tools
  ▸ tools  0.2 ms
    writes: answer=391, messages=[{"role": "tool", "content": "calculator(17*23) = 391"}], tools_used=["calculator"]
  ⟳ state changed: messages, tools_used, answer
    messages: [{"role": "user", "content": "What is 17*23?"}, {"role": "as… → [{"role": "user", "content": "What is 17*23?"}, {"role": "as…
    tools_used: [] → ["calculator"]
    answer:  → 391
  → edge tools → agent
  ◆ checkpoint demo_0001_4e0d8d running next=agent

── superstep 2 ──────────────────────────────────────────────
  plan: agent
  ▸ agent  0.1 ms
    writes: messages=[{"role": "assistant", "content": "Now the word count. TOOL:…, steps=2
    ↳ tokens agent: 'Now the word count. TOOL: word_count | this sentence has nine words in it'
  ⟳ state changed: messages, steps
    messages: [{"role": "user", "content": "What is 17*23?"}, {"role": "as… → [{"role": "user", "content": "What is 17*23?"}, {"role": "as…
    steps: 1 → 2
  ⑂ route after agent via route_after_agent()
    candidates: tools | __end__
    returned:   'tools'
    selected:   tools
  ◆ checkpoint demo_0002_caa18f running next=tools

── superstep 3 ──────────────────────────────────────────────
  plan: tools
  ▸ tools  0.0 ms
    writes: answer=7, messages=[{"role": "tool", "content": "word_count(this sentence has n…, tools_used=["word_count"]
  ⟳ state changed: messages, tools_used, answer
    messages: [{"role": "user", "content": "What is 17*23?"}, {"role": "as… → [{"role": "user", "content": "What is 17*23?"}, {"role": "as…
    tools_used: ["calculator"] → ["calculator", "word_count"]
    answer: 391 → 7
  → edge tools → agent
  ◆ checkpoint demo_0003_06131d running next=agent

── superstep 4 ──────────────────────────────────────────────
  plan: agent
  ▸ agent  0.1 ms
    writes: messages=[{"role": "assistant", "content": "FINAL: 17*23 is 391, and …, steps=3
    ↳ tokens agent: 'FINAL: 17*23 is 391, and the sentence has 9 words.'
  ⟳ state changed: messages, steps
    messages: [{"role": "user", "content": "What is 17*23?"}, {"role": "as… → [{"role": "user", "content": "What is 17*23?"}, {"role": "as…
    steps: 2 → 3
  ⑂ route after agent via route_after_agent()
    candidates: tools | __end__
    returned:   '__end__'
    selected:   __end__
  ◆ checkpoint demo_0004_425dc5 running next=<end>
  ◆ checkpoint demo_0004_425dc5 complete next=<end>

── superstep 5 ──────────────────────────────────────────────

■ run end: complete  supersteps=5 ckpt=demo_0004_425dc5
  final state:
    answer = 7
    messages = [{"role": "user", "content": "What is 17*23?"}, {"role": "assistant", "content":…
    question = What is 17*23?
    steps = 3
    tools_used = ["calculator", "word_count"]
```

### 怎么读这段输出

**① 开头五行的 run 头（`▶`）**

```
▶ run r_c22281d6  thread=demo
  graph: tool-loop-agent  entry: agent
  nodes: agent, tools
  channels: question[overwrite], messages[add], tools_used[add_unique], answer[overwrite], steps[overwrite]
  inputs: answer=, messages=[{"role": "user", "content": "What is 17…, question=What is 17*23?, steps=0, tools_used=[]
```

这是 `run_start` 事件，里面塞了整张图的**静态快照**。所以光看这一份 trace 就能重建图，不需要去读 `examples/agent.py`。

- `graph:` 名字 + 入口节点
- `nodes:` 全部节点
- `channels:` **每一列的 reducer 类型**——这是最该先看的一行。`question[overwrite]` 表示后写覆盖先写；`messages[add]` 表示每次写都是往列表末尾追加；`tools_used[add_unique]` 表示追加时去重。后面 `⟳` 段的 diff 长什么样，完全由这一行决定。
- `inputs:` 初始状态。注意显示被截断了（`…`），完整值在 JSONL 里。

**② `── superstep N ──` 一个 BSP 步**

整个执行模型是 Pregel/BSP 式的：一个 superstep 里并发的节点全部跑完、写进同一个 patch 集合、一起经 reducer 合并，然后才决定去向。所以步号是**调度屏障**，不是「第几次循环」。

`plan: agent` 是这一步打算跑哪些节点。只有一个说明这一步没并发。

**③ `▸ <node>  <耗时>` 节点开始**

`▸ agent  0.0 ms` —— 节点名 + 本次耗时。mock 场景下都是 0.0/0.1 ms。

**④ `writes:` 该节点返回的 patch**

```
writes: messages=[{"role": "assistant", ...}], steps=1
```

**这是 patch，不是完整状态。** 节点只声明自己改了哪些 channel。你看到的就是它 `return` 的那个 dict。这正是原则一的落点：节点没有能力偷偷改状态，它只能交一份 patch 出来。

**⑤ `↳ tokens` 流式输出预览**

```
↳ tokens agent: 'Let me compute that. TOOL: calculator | 17*23'
```

节点在一次 LLM 调用里产生的所有 token 被拼起来，在 `node_end` 时 flush 成一行。真实 trace 里这是 16 条 `node_token` 事件，这里聚合成人读的一行。这正是原则三——文本已经在 trace 里了，回放不用再问模型一遍。

不想要这行就 `--no-tokens`。

**⑥ `⟳ state changed:` reducer 生效的证据**

```
⟳ state changed: messages, steps
  steps: 0 → 1
  messages: [...] → [...]
```

两条信息：

- 冒号后那串是**经 reducer 合并后真正发生了变化的 channel**。注意节点 `writes:` 里写的东西不一定都会出现在这里——如果 reducer 判定值没变（比如 `add` 进去发现重复），它就不算「changed」。
- 缩进的 `before → after` 是每个变化 channel 的 diff。这是理解 reducer 最快的入口：`steps` 从 `0` 变成 `1` 是 `overwrite`；`messages` 两边都以 `[{"role": "user"...` 开头、右边更长，是因为 `add` 在追加。

不想要 diff 行就 `--no-diff`。

**⑦ `⑂ route after ... via ...()` 路由的完整决策**

```
⑂ route after agent via route_after_agent()
  candidates: tools | __end__
  returned:   'tools'
  selected:   tools
```

这是本项目最想看的一段。绝大多数框架只会告诉你「去了 tools」，lens 会告诉你**三个值**：

- `candidates` — 这个条件边**全部**的可能去向
- `returned` — 路由器函数实际 return 的字符串
- `selected` — 经 mapping 翻译后真正的目标节点

注意 `returned` 和 `selected` 在这个示例里**恰好同名**（mapping 是恒等映射），所以看不出区别。换 `approval` 那个例子就一目了然了：

```
⑂ route after plan via route_after_plan()
  candidates: approval | execute
  returned:   'approval'
  selected:   request_approval
```

路由器 return 的是 `'approval'`（一个抽象分支名），mapping 把它翻译成真正的节点 `request_approval`。**这三个值一起看，路由就不再是黑盒了**：你能分清「路由器想干什么」和「图把它接到了哪里」——这两件事出分歧的时候，bug 就在 mapping 里。

（`candidates` 只是名字列表；每个候选的人读说明在 `route_decision.descriptions` 里，`show` 目前没打印——这是本文「已知限制」第 3 条。）

**⑧ `→ edge X → Y` 无条件边**

```
→ edge tools → agent
```

和 `⑂` 相对：`tools` 没有条件，跑完必回 `agent`。`⑂` 是分叉，`→` 是直连。

**⑨ `◆ checkpoint <id> <status> next=...`**

```
◆ checkpoint demo_0004_425dc5 running next=<end>
◆ checkpoint demo_0004_425dc5 complete next=<end>
```

checkpoint id 格式是 `{thread_id}_{superstep:04d}_{uuid4 前 6 位 hex}`。

**同一个 id 出现两次是正常的**：`running` 是刚创建时落盘的那条，`complete` 是收尾时改写状态的同一条 checkpoint。不是两个 checkpoint，是一条被更新了。所以 `summary` 里 `checkpoint_created 6` 而唯一 id 只有 5 个。

`next=tools` / `next=<end>` 是恢复时该往哪走。

**⑩ 结尾那个「空」superstep，以及 `■ run end`**

```
── superstep 5 ──────────────────────────────────────────────

■ run end: complete  supersteps=5 ckpt=demo_0004_425dc5
```

这个 `── superstep 5 ──` 是**渲染器的产物，不是真的跑了一步**。查 trace 原始数据：`superstep_start` 的 `index` 只有 `[0, 1, 2, 3, 4]`，没有 5；是 `run_end` 事件本身带 `superstep=5`，渲染器给任何 superstep 变化都打横幅，于是多出一个空段。别被它误导成「多跑了一步」。

`supersteps=5`：注意这个字段在**中断/失败**时语义不同（那时它是出事那一步的索引，不是「跑了几步」），详见 [DESIGN.md](DESIGN.md) 第 11.6 节。

最后 `final state:` 是终态**摘要**（长值被截断，并且隐藏了 `__` 开头的内部 channel）。要完整状态请用 `state` 子命令或直接读 `checkpoints/*.json`。不想要这段就 `--no-state`。

### 换一个例子：interrupt / resume

```bash
py -3.13 -c "import examples.approval as a; a.build()"
```

跑完打印：

```
[1] first run status=interrupted supersteps=1
[2] paused at checkpoint approval-demo_0001_b1ff4a
    node=request_approval payload={'question': 'Approve this action?', 'action': 'delete_file', 'args': {'path': 'reports/2025-q4.csv'}, 'reason': 'stale export blocks the nightly rebuild'}
[3] human replies: decision='approve'
[4] resumed status=complete result="executed delete_file with {'path': 'reports/2025-q4.csv'}"

[trace written to traces\r_a0567c7f.jsonl]
[final] status=complete log=[...]
```

`invoke` 和 `resume` 两段写在**同一个 run_id** 的同一份 trace 里，所以 `show` 会把它们渲染成一个连续的故事。（上面的 checkpoint id 和 run_id 每次跑都不一样。）你会额外看到两个符号：

```
  ⏸ interrupt at request_approval payload={"question": "Approve this action?", ...}
```

```
■ run end: interrupted  supersteps=1 ckpt=approval-demo_0001_b1ff4a
  ...
  ⟳ state · resume payload changed: decision
```

`⏸` 是节点写入了保留 channel `__interrupt__`，运行在这里停下。中间那个 `■ run end: interrupted` 和随后那句 `⟳ state · resume payload changed:` 说明**下面接着的是 resume 段**：同一份 trace 里，第一段的 `run_end` 后面不是新 run，而是人类那一笔贡献（`decision` channel 被填上）之后继续跑。这是这份 trace 最值得看的地方——「暂停」和「继续」在同一根时间线上。

---

## 其他视图

### `summary` —— 按 kind 数事件

```console
$ py -3.13 -m lens_harness summary --no-color traces/r_c22281d6.jsonl
event summary
  node_token          16
  checkpoint_created  6
  node_end            5
  node_start          5
  route_decision      5
  state_merge         5
  superstep_start     5
  run_end             1
  run_start           1
```

对上一步的讲读：`superstep_start 5` = 5 个 superstep（0..4，第 5 个是空的）；`node_start/node_end 5` = 5 次节点执行；`checkpoint_created 6` 但唯一 id 5 个（原因见 `◆` 那节）；`node_token 16` 是 **3 次 LLM 调用**被 `chunk_size=12` 切出来的块数总和（superstep 0 / 2 / 4 各 4 / 7 / 5 块）。

这个视图最实用的场景是**对比两份 trace**：跑同一个问题两次，`summary` 一变就知道哪里不一样了。

### `checkpoints` —— 列表（已按 id 去重）

```console
$ py -3.13 -m lens_harness checkpoints --no-color traces/r_c22281d6.jsonl
demo_0000_51d41f  ss=0   running      next=tools
demo_0001_4e0d8d  ss=1   running      next=agent
demo_0002_caa18f  ss=2   running      next=tools
demo_0003_06131d  ss=3   running      next=agent
demo_0004_425dc5  ss=4   complete     next=end
```

同一个 id 只显示最新状态（`list_checkpoints` 去重），所以是 5 行不是 6 行。

### `state` —— 某个 checkpoint 的状态

```console
$ py -3.13 -m lens_harness state --no-color traces/r_c22281d6.jsonl --at 2
```

```json
{
  "question": "What is 17*23?",
  "messages": [
    {"role": "user", "content": "What is 17*23?"},
    {"role": "assistant", "content": "Let me compute that. TOOL: calculator | 17*23"},
    {"role": "tool", "content": "calculator(17*23) = 391"},
    {"role": "assistant", "content": "Now the word count. TOOL: word_count | this sentence has nine words in it"}
  ],
  "steps": 2,
  "answer": "391",
  "tools_used": ["calculator"]
}
```

为了省行数，上面把每个 `{"role": ...}` 对象压成了一行；真实输出是 `json.dumps(ensure_ascii=False, indent=2)`，嵌套对象各占多行。内容一字不差。

> 关于顺序：这里的 key 顺序来自 `summarize_state` 遍历 state 的插入顺序，实测**同一份 trace 反复 `state`、以及不同次运行，顺序都稳定**（6 次运行都是 `question, messages, steps, answer, tools_used`）。不稳定的只有 `show` 里 `⟳` 下面的 diff 行——那是另一处代码（见「已修复」里 diff 键序那条）。

> 关于完整性：`state` 只隐藏 `__` 开头的内部 channel（如 `__interrupt__`）。这个示例 trace 里本来就没有 `pending` 之类的 channel，所以看起来「全都在」。

⚠️ **重要**：这个视图重建的是**摘要**，不是完整状态。它取的是 `state_merge` 事件里那个经 `summarize_state` 截断过的字段。如果你的 channel 里塞了 500 字符的字符串或 40 项的列表，这里会看到 `…(500 chars)` / `…(+32 more)`。要**完整**状态请读 `traces/checkpoints/<id>.json`。原因见 [DESIGN.md](DESIGN.md) 第 11.4 节。

### `watch` —— 跟随正在写入的 trace

```bash
py -3.13 -m lens_harness watch traces/r_c22281d6.jsonl   # Ctrl-C 停止
```

先打一行 `following traces/<run_id>.jsonl (Ctrl-C to stop)`，然后每次新事件进来就清屏重渲染（发 `ESC[H` + `ESC[2J`）。**它只读已有的文件**，所以对一个已经跑完的 trace 用，看到的就是一遍静态重绘，然后**一直等在那里**（文件不会再有新行），需要 Ctrl-C 停止。

（`cmd_watch` 捕获 `KeyboardInterrupt` 后 `return 0`。注意通过 `py` 启动器发信号时，实测外层收到的是 130 而不是 0——如果你在脚本里依赖这个退出码，别依赖它。）

### `serve` —— 在浏览器里看同一份 trace

```bash
py -3.13 -m lens_harness serve traces/r_c22281d6.jsonl
```

一行起服务，然后打开它打印的地址：

```
lens serving traces/r_c22281d6.jsonl
  mode : replay (finished trace)
  open : http://127.0.0.1:8000/
  stop : Ctrl-C
```

页面长这样（截图是真跑出来的，不是在文档里画的）：

| 区域 | 内容 |
| --- | --- |
| 顶栏 | 图名、run id、连接状态、run 状态、`N / M events` |
| 左：GRAPH | 从 `run_start.data.graph` 现算的图画布，含访问次数角标、当前节点脉冲、走过的边高亮 |
| 右：INSPECTOR | 五个标签页：`state` / `diff` / `writes` / `events` / `checkpoints` |
| 底：时间轴 | 每个 superstep 一个块（点了跳过去）、拖动条、播放/上一步/下一步/回到最新 |

页面上能做的时间旅行是真的：

- **拖回去**：视图立刻变成「只重放前 N 条事件」的样子——不是撤销，是用更短的前缀重算。所以拖到任意位置都自洽，包括拖到 `run_end` 之前。
- **播放**：`▶` 从当前位置每 90 ms 推进一步，正好是「让 trace 重新发生一遍」。
- **键盘**：`←` / `→` 单步，`空格` 播放/暂停，`l` 跳到最新，`Home` / `End` 两头。

看一段**正在跑**的 trace：

```bash
py -3.13 -m lens_harness serve traces/<run_id>.jsonl --follow
```

`--follow` 下服务端不停在 EOF，每 100 ms 重读文件，新事件一落盘就推过去。顶栏的连接状态会显示 `live`。

几个细节：

- `--port 0` 让系统挑一个空闲端口（打印出来的地址里是真实端口）。
- 默认只绑 `127.0.0.1`。绑 `0.0.0.0` 会把你的 trace 发布到网络上，程序会打一行警告。
- 刷新页面不会重复事件：`seq` 稠密且单调，浏览器重连带 `Last-Event-ID`，服务端从下一条续传。
- 页面是**只读**的：不能跑图、不能 resume、不能提交审批。它只有一个方向，trace → 屏幕。

它为什么这么设计（为什么是 SSE 不是 WebSocket、图和边怎么布线、页面怎么保证不 import 内核），见 [DESIGN.md](DESIGN.md) 第 12 节。

---

## LLM 支持

内核不绑定任何模型。`lens_harness/llm.py` 里有三个实现：

| 类 | 用途 |
| --- | --- |
| `MockLLM` | 确定性假模型。三种模式按顺序匹配：script 规则 → `replies` FIFO → `default`。`chunk_size=12` 模拟流式。默认 `auto` 无 key 时用它 |
| `OpenAICompatibleLLM` | 任何 OpenAI 兼容端点（OpenAI / 本地 vLLM / Ollama / DeepSeek …） |
| `RecordingLLM` | 包一层，按 prompt hash 缓存到 `llm_cache/`，便于复盘 |

### spec 选择

`load_llm(spec)` 的优先级：**函数参数 > `LENS_LLM` 环境变量 > `"mock"`**。

支持三种写法：

```
mock                   纯假模型，default 回复 "ok"
auto                   有 key 用真模型，没 key 自动回落到 MockLLM
openai:<model>         强制 OpenAI 兼容端点，如 openai:gpt-4o-mini
```

写成别的会直接报错：

```
ValueError: unknown LLM spec 'bogus'; use mock | auto | openai:<model>
```

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LENS_LLM` | `mock` | 默认 spec（命令行 `--llm` 会覆盖它） |
| `LENS_LLM_API_KEY` | — | 优先读它 |
| `OPENAI_API_KEY` | — | 上一条没有时读它 |
| `LENS_LLM_BASE_URL` | `https://api.openai.com/v1` | 换端点时用 |
| `LENS_LLM_MODEL` | `gpt-4o-mini` | `auto` 分支的默认模型 |

丢 key 时的报错很明确：

```
RuntimeError: no API key: set LENS_LLM_API_KEY (or OPENAI_API_KEY), or use MockLLM instead
```

接自建端点：

```bash
export LENS_LLM_API_KEY=sk-...
export LENS_LLM_BASE_URL=http://127.0.0.1:8000/v1
export LENS_LLM_MODEL=qwen2.5-7b-instruct
py -3.13 -m lens_harness run examples.agent:build --llm auto
```

### ⚠️ `run` 的 `--llm` 会改变示例的行为

这一点容易踩：`examples/agent.py` 的 `build()` 在**不传** `llm_spec` 时用的是**自带的 3 条 scripted replies**（所以会跑满 5 个 superstep，算出 391 和 7）；一旦你传 `--llm mock` 或 `--llm auto`，它就走 `load_llm()`，得到的是一个 `default="ok"` 的纯 MockLLM，于是**只跑 1 个 superstep**就结束了：

```console
$ py -3.13 -m lens_harness run examples.agent:build --input '{"question":"What is 17*23?"}'
...
■ run end: complete  supersteps=5

$ py -3.13 -m lens_harness run examples.agent:build --llm mock --input '{"question":"hi"}'
...
    answer =
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    steps = 1
```

两个都是「对」的，只是跑的是不同脚本。想看完整的 5 步故事，**不要传 `--llm`**。

---

## trace 对比（M3）

README 早前说过「没有内建的 diff 命令」，M3 补上了。核心问题：两份 trace 的 `ts` / `run_id` / `checkpoint_id` / `parent_id` / `duration_ms` 这 5 个字段**每次运行必然不同**，裸 `diff` 永远非空、永远没有信息量。`lens diff` 做的是：把这层噪音归一化掉之后，剩下的差异就是**行为差异**。

### 终端

```bash
py -3.13 -m lens_harness diff traces/run-a.jsonl traces/run-b.jsonl
```

同一个示例跑两次，输出是一行：

```
■ identical: 36 aligned records, every behavioral field matches
```

（退出码 0。）换一个问题再跑，输出分三段：

```
trace diff  36 aligned records
  same: 24   changed: 12   only-a: 0   only-b: 0

── changed ──
  run_start 0  inputs
    a: {"answer": "", "messages": [{"content": "What is 17*23?"...
    b: {"answer": "", "messages": [{"content": "hi there"...

── only-b (only in trace b) ──
  node_start 2 extra ...
```

退出码学 GNU diff：**有差异 = 1**（是信号，不是报错），脚本里写 `lens diff a b || true`。

### 归一化规则（`lens_harness/compare.py`）

| 字段 | 处理 | 理由 |
| --- | --- | --- |
| `ts` / `run_id` / `duration_ms` | 丢弃 | 身份与时序不是行为 |
| `checkpoint_id` / `parent_id` | 替换成 `ss<N>` 占位符 | uuid 后缀每次都变，但「第 N 步的 checkpoint」是故事里稳定的位置 |
| `node_token` 事件 | 聚合成每 `(superstep, node)` 一条整段文本 | 逐 chunk 对齐只会 diff 出「切块方式」，不是内容 |

其余逐字段保留并对齐：节点事件按 `(kind, superstep, node)` 配对，`superstep_start` 按 index，含 resume 的 trace（会有两条 `run_start`）按出现顺序配对。对齐后四类：`same` / `changed`（同位置不同值，列出具体字段）/ `only-a` / `only-b`。

**判读规则**：`only-a`/`only-b` 里的节点级事件是真差异（一条路由没走、一步没跑）；但 run 结束位置不同时，`run_end` 和终态 checkpoint 会因 superstep 号对不上而各自出现在 only 侧——这是结构性余波，不是多跑了一个节点。

### 浏览器

```bash
py -3.13 -m lens_harness serve traces/run-a.jsonl traces/run-b.jsonl
```

服务端多开两个路由：`/compare`（对比页，单文件、零依赖，与主页面同一套纯度标准——不 import 内核）和 `/api/diff`（对比结果 JSON，每次请求现算，不用缓存）。页面按 `changed` / `only-a` / `only-b` 分节渲染。单 trace 启动时 `/compare` 返回 404、`/api/diff` 返回 504，都不是静默。

### 测试

`tests/test_compare.py`（8 条，`py -3.13 -m tests.test_compare`）。最要紧的一条是 `test_identical_runs_diff_clean`：**同一个图跑两次，diff 必须为空**——这是归一化的验收标准，也是「diff 非空 ⟹ 行为有差」这条推理链的地基。另有一条故意钉住 only 侧的结构性余波（见上面判读规则），防止将来有人把它当 bug「修」掉。


### 怎么找到我刚刚跑的那份 trace？

不要用 `ls -t`。本机 `ls` 是 WinuxCmd 1.2.1，`-t` **不按 mtime 排序**（实测过，给的是错的顺序）。用 Python：

```bash
py -3.13 -c "import glob,os;print(max(glob.glob('traces/*.jsonl'),key=os.path.getmtime))"
```

想要不带路径的文件名，加 `os.path.basename`。或者干脆自己指定 `--trace-dir`：

```bash
py -3.13 -m lens_harness run examples.agent:build --trace-dir mytraces
```

（`--trace-dir` 会顺便建目录，并在里面再建一个 `checkpoints/`。相对路径就是相对当前目录。）

### 我有两份 trace，想看 diff

目前**没有内建的 diff 命令**（`cli.py` 只有那八个子命令）。可行的手法：

```bash
# 同一个示例跑两遍，拿到两份 trace 的路径（各自落在独立目录，不互相覆盖）
T1=$(py -3.13 -c "import examples.agent as a,glob; a.build(trace_dir='traces/run-a'); print(glob.glob('traces/run-a/*.jsonl')[0])" | tail -1)
T2=$(py -3.13 -c "import examples.agent as a,glob; a.build(trace_dir='traces/run-b'); print(glob.glob('traces/run-b/*.jsonl')[0])" | tail -1)

py -3.13 -m lens_harness show --no-color "$T1" > /tmp/a.txt
py -3.13 -m lens_harness show --no-color "$T2" > /tmp/b.txt
diff /tmp/a.txt /tmp/b.txt
```

比之前先看 `summary` / `checkpoints`，能一眼看出是不是「事件数就不一样」这种粗差。

diff 之前先知道**哪些差异是噪音**：`ts`、`run_id`、`data.checkpoint_id`、`data.parent_id`、`data.duration_ms` 这 5 个字段每次运行都不同，所以两份 trace 的 `diff` 一定**不是空的**。渲染成文本后它们只会落在三种行上：

- `▶ run r_xxxxxxxx  thread=demo` —— run_id
- 任何含 checkpoint id 的行：`◆ checkpoint demo_0000_xxxxxx ...`、`■ run end: complete  supersteps=5 ckpt=demo_0004_xxxxxx`
- `▸ <node name>  0.1 ms` —— 耗时（节点名每次相同，只有毫秒数会变）

实测（示例跑两遍再 diff）：带 `<` / `>` 的内容行十几行，**全部**属于这三类。其中稳定的是 2 条 `▶ run` + 2 条 `■ run end` + 12 条 `◆ checkpoint`（每边 5 个 running + 1 个 complete）；剩下几条是 `▸ <node>  x.x ms`，数量随「两次运行里有几个节点的耗时恰好相同」浮动——相同的那些不进 diff。

判读规则就一句话：把这三类行过滤掉，`diff` 还剩东西，那就是真的行为差异——详见下面「已修复」里 diff 键序那条。

> 另外注意：`diff` 在「两份文件不同」时退出码是 **1**，这是它正常的「有差异」信号，不是命令失败。想让它别在脚本里看起来像报错，写成 `diff a b || true`。

### 为什么 `writes:` 和 `⟳ state changed:` 列的东西不一样？

`writes:` 是节点**交上来的 patch**，`⟳` 是**reducer 合并后真正变化的 channel**。reducer 有可能判定「这次写进去等于没变」（比如 `add_unique` 遇到重复项），那它就不会出现在 `⟳` 里。这两个视图一起看，才能确认 reducer 是不是按你以为的方式在工作。

### `summary` 里 `checkpoint_created` 为什么比 checkpoint 个数多？

同一个 checkpoint 会发两次事件：创建时（`running`）和终态时（`complete`/`interrupted`）。`summary` 数的是**事件**，`checkpoints` 数的是**去重后的 checkpoint**。示例里就是 6 vs 5。

### 中文 / emoji 打印会崩吗？

不会。见开头「关于中文和 emoji 会不会崩」。cli 已强制 UTF-8。

### 能建 venv 吗？

不要。本机 `ensurepip` 失败，而且这个项目**零依赖**，没有建 venv 的任何理由。直接 `py -3.13` 用系统解释器。

---

## 已知限制（如实列出）

写这份 README 时逐条实测出来的，不粉饰：

1. **每次运行的 `run_id` 和 checkpoint id 都不同。** 本文样例里的 `r_c22281d6` / `demo_0000_51d41f` 这些照抄是跑不出来的，按上面「怎么找到我刚刚跑的那份 trace」自己取。

2. **`state` 给的是摘要不是完整状态。** M3 之后 CLI 会先读 `checkpoints/<id>.json` 给完整状态（输出首行注明来源），只有 checkpoint 文件缺失时才回退到 trace 摘要。直接调 `replay_state_summary_at` 得到的仍是摘要。

3. **`show` 不打印路由候选的人读说明。** `route_decision` 事件里其实带了 `descriptions`（比如 `'approval': 'the action is destructive and needs a human'`），但终端渲染器只画了 `candidates` 名字。想看说明得读 JSONL，或者去浏览器 lens 里把鼠标停在虚线上（前提是这份 trace 的图快照带 `mapping`+`descriptions`，新跑出来的都带；早期写下的 trace 里 `descriptions` 存的是节点名，页面会自动兼容，只是没有说明可显示）。

4. **浏览器 lens 是只读的，且一次只看一份 trace。** 页面只做 `trace → 屏幕` 这一个方向：不能跑图、不能 resume、不能提交审批、没有 `POST`（服务端只实现了 `do_GET`）。也没有「开两个标签对比两份 trace」这种功能。它存在的意义是「同一份事件换一种眼睛看」，不是做一个工作台。要是你正好需要工作台，那这个项目没打算做（见第 10 节）。

5. 更完整的设计层自查清单在 [DESIGN.md](DESIGN.md) 第 11 节（其中一部分已在后续修复）。那些是设计落差，不影响日常使用，但值得读。

### 已修复（本文早期版本记录过，现不成立）

诚实起见保留这一段，免得你看到旧版 README 的描述对不上：

- **`run` 曾只能跑签名匹配的示例。** 早期 `cmd_run` 无条件传 `llm_spec=`，导致 `run examples.approval:build` 抛 `TypeError: build() got an unexpected keyword argument 'llm_spec'`。现在改为用 `inspect.signature()` 检查目标声明了哪些参数，只传它接受的（`inputs` / `trace_dir` / `llm_spec` 三个候选）。所以 `run` 对任何 `build()` 都能用，不同示例可以有不同签名。
- **`state` 的 docstring 曾与实际参数顺序相反。** 现已重写，列全 7 个子命令，并明确 `state` 是「trace 在前、checkpoint id 在后」或 `--at N`。以 `--help` 为准，两者现在一致。
- **`⟳` 下的 diff 行序曾经不稳定。** 早期 `State.diff()` 遍历 `set(self.values) | set(other.values)`，顺序随 `PYTHONHASHSEED` 变，而这个 diff 会被序列化进 trace，于是同样输入的不同次运行产出「内容相同、字节不同」的 trace 文件。现在改为：已声明 channel 按 schema 顺序在前，其余（保留 channel）按字母序在后。
  实测连续跑 6 次，superstep 0 的 diff 键序稳定为 `['messages', 'steps']`、superstep 1 稳定为 `['messages', 'tools_used', 'answer']`。
  好处是**行为可复现**，但不是**文件逐字节相同**——这点要说准，两种说法差很远：

  同样输入连跑 3 次，事件条数与每条的字段集合完全一致（`49` 条、`seq` 稠密），`⟳ state changed:` 下的 diff 键序也稳定；但 `md5sum` 三次都不同。实测把这 3 份 trace 按事件对齐后逐字段比较，只有 5 个字段会变：

  ```
  ts                   每个事件的时间戳
  run_id               每次运行新建
  data.checkpoint_id   格式 {thread}_{superstep:04d}_{uuid4前6位hex}
  data.parent_id       指向上一轮的 checkpoint id
  data.duration_ms     节点真实耗时
  ```

  把这 5 个字段抹掉再比对，3 份 trace 逐事件字段完全相同（`seq` 也稠密、同为 49 条）。渲染出来也一样：`show` 输出只有 run id、checkpoint id、`0.1 ms` 这类耗时不同，把它们归一化后三次逐行一致。

  实践含义：`diff` 两份 trace **不会**是空的，但差异只该落在那 5 个字段上。如果 diff 里出现了别的行（事件顺序、`changed_channels`、diff 键序、`writes` 内容），那就是真的行为差异，值得追。

## 延伸阅读

- [DESIGN.md](DESIGN.md) —— 12 章完整设计：定位、两层架构、事件模型、BSP superstep、Reducer 语义、Checkpoint DAG、路由透明性、时间旅行、与 LangGraph 的对照、明确不做的清单、已知落差自查、浏览器 lens（第 12 章）。
- `lens_harness/__init__.py` 的 docstring —— 建议的源码阅读顺序。
- `tests/test_kernel.py` —— 25 条测试就是 25 条契约，比文档更准。特别推荐 `test_parallel_writes_accumulate_not_overwrite`（并行写经 reducer 累加而不是互相覆盖）、`test_every_event_validates_and_seq_is_dense`（`seq` 必须是 `0..n-1` 稠密）、`test_replay_state_at_rebuilds_state_from_trace_alone`（回放不需要重跑）。
- `tests/test_serve.py` —— 31 条传输层契约。最值得看的是「页面纯度」那几条：它们用最笨的办法（在 HTML 文本里 grep `lens_harness` / `runtime.py` / `state_merge(`）来确保渲染器没有偷偷 import 内核。如果有天有人为了图方便在页面里直接算 state 增量，这几条测试会红。还有一条钉的是「候选值要经 `mapping` 解析」——那个错误在历史 trace 上目视看不出来，只会在新 trace 上把边弄丢。最后六条是传输层硬化留下的（DESIGN 12.9）：流必须在 `done` 之后真的关掉 socket、关掉的标签页不能留下生产线程、trace 被重写后要能重新读、截断必须报错而**不能**发 `done`、坏游标回 400、目录当 trace 传要当场拒绝——其中第一条必须绕开 `urllib` 开原始 socket，否则它永远看不到那个 bug。
