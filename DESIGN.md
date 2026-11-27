# lens-harness 设计文档

一个可读的、LangGraph 风格的编排内核，配一份可以回放、可以 diff、可以时间旅行的 trace。

这份文档只描述**已经实现**的行为。每一条断言都能在代码里找到出处，绝大多数直接标了文件与行号。凡是「代码实际行为与直觉/注释/任务描述不一致」的地方，都在第 11 节逐条列出，不做美化。

阅读顺序（也是 `lens_harness/__init__.py:4-13` 里写的顺序）：

| 顺序 | 文件 | 它是什么 |
| --- | --- | --- |
| 1 | `lens_harness/events.py` | trace 契约，唯一事实来源 |
| 2 | `lens_harness/state.py` | channel、reducer、patch |
| 3 | `lens_harness/graph.py` | 图声明与编译 |
| 4 | `lens_harness/runtime.py` | superstep 主循环（核心） |
| 5 | `lens_harness/checkpoint.py` | 撑起时间旅行的 DAG |
| 6 | `lens_harness/render.py` | 一个渲染器，即 trace 的*消费者* |
| 7 | `lens_harness/serve.py` | 把同一份 trace 推给浏览器（第 12 节） |
| 8 | `lens_harness/web/index.html` | 浏览器渲染器本身，trace 的另一个消费者 |

---

## 1. 项目定位与三大原则

lens-harness 不是「又一个 agent 框架」，而是**把 agent 框架内部那层看不见的调度摊开给你看**的实验装置。范例图 `examples/agent.py` 画的就是每个框架都藏着的那条链路：

```
agent ──(tool_calls?)──▶ tools ──▶ agent ──(done?)──▶ END
```

围绕这条链路，整个内核只坚持三件事。

### 原则一：Trace 是唯一事实来源

运行过程中发生的每一个可观察动作，都被写成一个 `Event`：

```python
@dataclass(frozen=True, slots=True)
class Event:
    seq: int          # 全局递增，0..n-1 无空洞
    ts: float
    run_id: str
    thread_id: str
    superstep: int
    kind: str
    data: dict[str, Any]
```

（`lens_harness/events.py:71-81`）

关键点是**没有任何旁路**：状态在哪、谁写了什么、路由为什么这样选、checkpoint 何时落盘，全部只通过事件表达。测试 `test_every_event_validates_and_seq_is_dense`（`tests/test_kernel.py:316`）就在守这条线——`seq` 必须是 `0..n-1` 的稠密序列，否则下游所有消费方的排序都不可信。

### 原则二：UI 是 trace 的纯函数

`lens_harness/render.py:1-9` 的开头就写死了这条约束：渲染器**只吃 `list[Event]`**，从不 import `graph` 或 `runtime`。核对 import 语句可以确认——`render.py:18-32` 里只有 `from .events import ...`，没有别的内核模块。

这不是洁癖，是验收标准：如果渲染器需要偷看内核内部状态才能画出正确的东西，那说明 trace 契约漏了信息。

### 原则三：回放不需要重跑 LLM

`node_token` 事件把 LLM 的流式输出逐块记进 trace（`lens_harness/runtime.py:356`）：

```python
emit_token=lambda tok: emitter.emit(NODE_TOKEN, node=name, token=tok),
```

配合 `MockLLM` 的确定性（同样的 prompt 必得同样的回复，`lens_harness/llm.py:87-93`），一份 trace 可以离线重放、可以 diff 两次运行、可以在没有网络的情况下看完整推理过程。

### 判断标准：不帮你看清事情的功能不做

代码里有一条可验证的自治标准——**如果一个功能不能帮你看清某件事，就不做**。它对内体现为「能少一个模块就少一个模块」，对外体现为第 10 节的「明确不做清单」。

---

## 2. 两层架构

```
┌─ kernel: lens_harness/ ────────────────┐      ┌─ lens: 只读渲染 ─────────┐
│ events  state  graph  runtime          │      │ render.py（终端渲染器）   │
│ checkpoint  llm                        │ ───▶ │ serve.py + web/（浏览器） │
│         │                              │trace │                          │
│         └── Emitter → EventSink ───────┘      │ 两者都只吃 list[Event]    │
└────────────────────────────────────────┘      └──────────────────────────┘
```

两个渲染器是**两个并列的消费者**，不是一条流水线：`serve.py` 不经过 `render.py`，`render.py` 也不知道 `serve.py` 存在。它们的共同点只有一处——都不 import 内核（见第 12 节）。

### kernel 层（`lens_harness/`）

十个文件，各管一件事，没有一层是「框架必备但没人读」的：

| 文件 | 职责 | 关键出口 |
| --- | --- | --- |
| `events.py` | 事件契约、schema 校验、sink、Emitter、trace 读写 | `Event`, `EVENT_SCHEMA`, `Emitter`, `JsonlSink` |
| `state.py` | channel 声明、4 个内置 reducer、`apply_patch` | `State`, `Channel`, `add`/`add_unique`/`merge_dict`/`overwrite` |
| `graph.py` | `StateGraph` 声明 + `compile()` 校验 | `StateGraph`, `CompiledGraph`, `START`, `END` |
| `runtime.py` | BSP 主循环 | `Runtime.invoke` / `Runtime.resume` / `_drive` |
| `checkpoint.py` | 文件后端 checkpoint store + DAG 查询 | `CheckpointStore`, `Checkpoint` |
| `llm.py` | LLM 边界：Mock / Recording / OpenAI 兼容 | `load_llm`, `MockLLM`, `LLMResponse` |
| `render.py` | 终端渲染器（trace 消费者） | `render_events`, `render_checkpoint_dag`, `replay_state_at` |
| `cli.py` | `python -m lens_harness` 子命令 | `main`, `build_parser` |
| `__init__.py` | 收窄的公共面（`__all__`） | — |
| `__main__.py` | 模块入口（6 行） | — |

`__init__.py:51-67` 的 `__all__` 是刻意的窄口：40 个名字、分 5 组（graph / state / runtime / checkpoints / events / llm），读完就能知道这个项目是什么形状——没有 `_private`、没有内部状态容器、没有 sink 实现之外的扩展点。

### lens 层

`lens` 层有两个并列的只读消费者，都只吃 `list[Event]`：

**终端渲染器 `lens_harness/render.py`**，提供四种视图：

- `render_events(events, options=..., palette=...)` —— 主视图，按 superstep 分组（`render.py:94-259`）
- `render_checkpoint_dag(events)` —— checkpoint 结构的 ASCII 树（`render.py:265-292`）
- `render_summary(events)` —— 每种事件的计数（`render.py:295-305`）
- `replay_state_at(events, checkpoint_id)` —— 只从 trace 重建状态（`render.py:308-332`，注意第 11 节第 4 条的落差）

**浏览器渲染器 `lens_harness/serve.py` + `lens_harness/web/index.html`**，见第 12 节：

- `serve.py` 把同一份 trace 用 SSE（`text/event-stream`）推给浏览器，纯标准库（`http.server.ThreadingHTTPServer`）
- `web/index.html` 是一个自包含单文件页面：图画布、superstep 时间轴、五个检查器面板、拖动条
- 页面的全部状态由一个纯函数 `derive(events, upto)` 算出，`paint()` 只负责把它写进 DOM

`lens` 层不做的事同样重要：它不执行图、不加载 checkpoint 文件、不读环境变量里的 key。`serve.py` 也不发起任何 LLM 调用——它连 `llm` 模块都没 import。

### 为什么不追求生产稳定性 / 性能 / 分布式

这是**学习用途**的项目，取舍是明写出来的：

- **不追求生产稳定性**：`Runtime` 用线程池并发（`runtime.py:332`），没有背压、没有超时、没有重试；节点抛异常被当成数据捕获（`runtime.py:366-372`），不是崩溃。
- **不追求性能**：checkpoint 是一个 checkpoint 一个 JSON 文件、缩进 2 空格（`checkpoint.py:104`）；`CheckpointStore.all()` 每次都扫目录重新解析（`checkpoint.py:144-157`）。模块注释直接承认这件事：「Storage is deliberately dumb: one JSON file per checkpoint plus an index file. A learning tool should let you `cat` a checkpoint.」（`checkpoint.py:8-9`）
- **不做分布式**：没有消息队列、没有 worker 池、没有跨进程调度。一个 `Runtime` 实例就是一个进程内的循环。

性能上唯一认真设计过的地方是**并发正确性**，因为那不是性能问题，而是会静默丢数据的正确性问题——见第 5 节。

---

## 3. 事件模型

`EVENT_SCHEMA`（`lens_harness/events.py:39-53`）是全部契约的源头，总共 11 种事件：

```python
EVENT_SCHEMA: dict[str, frozenset[str]] = {
    RUN_START: frozenset({"graph", "inputs"}),
    RUN_END: frozenset({"status", "supersteps", "final_state"}),
    SUPERSTEP_START: frozenset({"index", "nodes"}),
    NODE_START: frozenset({"node", "input"}),
    NODE_TOKEN: frozenset({"node", "token"}),
    NODE_END: frozenset({"node", "writes", "duration_ms"}),
    STATE_MERGE: frozenset({"patches", "changed_channels", "state"}),
    ROUTE_DECISION: frozenset({"source", "condition", "candidates", "returned", "selected"}),
    CHECKPOINT_CREATED: frozenset({"checkpoint_id", "parent_id", "next_nodes", "status"}),
    INTERRUPT: frozenset({"checkpoint_id", "node", "payload", "resume_schema"}),
    NODE_ERROR: frozenset({"node", "error_type", "error", "traceback"}),
}
```

`validate_event(kind, data)`（`events.py:60-68`）在**发出前**校验：少了必填键抛 `EventSchemaError`，未知 kind 也抛。所以内核写错字段是当场炸，而不是产出一份 lens 读不懂的 trace。

注意 schema 只约束**必填键的下限**，实际事件会带额外键（例如 `run_start` 还带 `config`，`run_end` 还带 `checkpoint_id`/`error`，`state_merge` 还带 `diff`/`origin`）。这在第 11 节有说明。

下面逐个说。

### 3.1 `run_start`

- **何时发出**：`Runtime.invoke` 一进来，在任何节点执行之前（`runtime.py:112-117`），一次 run 只一次。
- **必填 keys**：`graph`、`inputs`。实际还带 `config`（`thread_id`、`max_supersteps`）。
- **`graph` 是什么**：`StateGraph.snapshot()` 的产物（`graph.py:170-201`）——静态图的完整描述：`name` / `entry` / `nodes`（含 `description`、`tags`）/ `edges` / `conditional_edges` / `channels`（含每个 channel 的 reducer 名）。
- **`inputs` 是什么**：`summarize_state(inputs)`，**摘要**不是原值。
- **能回答什么问题**：这张图长什么样？入口是谁？每个 channel 用哪个 reducer？这次 run 的输入是什么？——lens 画静态图**只需要这一个事件**，永远不必 import 图定义代码。
- **注意**：`resume()` 不发这个事件（第 11 节第 3 条）。

### 3.2 `run_end`

- **何时发出**：`_drive` 退出前，`try/except` 之后（`runtime.py:291-298`），无论成功、失败还是中断都发。
- **必填 keys**：`status`、`supersteps`、`final_state`。实际还带 `checkpoint_id`（终止 checkpoint，可能为 `None`）与 `error`。
- **`status` 的取值**：`complete` / `failed` / `interrupted`（`checkpoint.py:26-28` 的常量）。
- **能回答什么问题**：这次跑成了还是挂了？跑了多少步？最终状态是什么？终止在哪个 checkpoint？

### 3.3 `superstep_start`

- **何时发出**：每个 superstep 的开头，`plan` 去重之后（`runtime.py:227-228`）。
- **必填 keys**：`index`（superstep 序号，从 0 开始）、`nodes`（本步要跑的节点名，已去重保序）。
- **能回答什么问题**：这一步打算跑谁？——这是交互式调试里最有用的一行，因为「计划跑的」和「实际跑了的」对照起来能立刻发现调度问题。

### 3.4 `node_start`

- **何时发出**：每个节点函数被调用前，`_run_node` 里第一件事（`runtime.py:349`）。
- **必填 keys**：`node`、`input`（节点的**入参快照** `summarize_state(snapshot)`）。
- **能回答什么问题**：这个节点看到的输入是什么？——同一步里并行的多个节点，它们的 `input` 应该逐一相等（同一份 pre-merge 快照，见第 4 节）。

### 3.5 `node_token`

- **何时发出**：节点通过 `ctx.emit_token(chunk)` 主动上报时（`runtime.py:356`），一个 chunk 一条。LLM 节点用它流式输出。
- **必填 keys**：`node`、`token`。
- **能回答什么问题**：模型到底一个字一个字地吐了什么？——渲染器会把同一节点的 token 缓冲起来，在 `node_end` 处合并成一行（`render.py:112-126`）。
- **可选性**：不带 LLM 的节点一条都不发，这是正常的。

### 3.6 `node_end`

- **何时发出**：节点函数正常返回后（`runtime.py:376-381`）。
- **必填 keys**：`node`、`writes`（节点返回的 patch，`summarize_state(patch or {})`）、`duration_ms`（保留 2 位小数）。
- **能回答什么问题**：这个节点写了哪些 channel？花了多久？——`writes` 是**合并前**的原始 patch，和后面的 `state_merge` 对照就能看出 reducer 做了什么。

### 3.7 `state_merge`

- **何时发出**：一个 superstep 里所有节点的 patch 都应用完之后（`runtime.py:412-429`）。
- **必填 keys**：`patches`、`changed_channels`、`state`。实际还带（两种来源不同，别指望字段齐全）：
  - **superstep 合并的那条**（`runtime.py:412-429`）带 `diff`：`{channel: {"before": ..., "after": ...}}`，逐 channel 的前后值。注意 `diff` **显式排除了所有 `__` 开头的 channel**（`runtime.py:421-428`），因为 `summarize_state` 会把它们渲染成 `null → null` 的无意义行。
  - **`resume()` 的那条**（`runtime.py:173-179`）带 `origin="resume_payload"`，**没有 `diff`**（它调用 `apply_patch` 后直接 emit，不经过 `_merge_and_route`）。
- **`changed_channels` 从哪来**：`State.apply_patch` 的返回值，只包含**真的变了**的 channel（`state.py:241-243`）。这就是为什么同一份 `state_merge` 里 `changed_channels` 常常比 `patches` 的键少。
- **能回答什么问题**：这一步到底改了什么？改前改后各是什么？——「这一步真的碰到了 messages 吗」是每次调试都会问的问题。

### 3.8 `route_decision`

- **何时发出**：`_merge_and_route` 里，对 `plan` 中每个节点调用 `next_targets` 之后（`runtime.py:431-446`）。静态边和条件边**都发**。
- **必填 keys**：`source`、`condition`、`candidates`、`returned`、`selected`。
- **条件边与静态边的差别**：
  - 条件边：`condition` 是路由器名字，`candidates` 是全部候选键，`returned` 是路由器**原始返回值**（可能是 str，也可能是 list），`selected` 是映射后的目标节点列表。还额外带 `descriptions`。
  - 静态边：`condition` 为 `None`，`candidates`/`returned`/`selected` 三者都等于边目标列表。实测形态：`{'source': 'tools', 'condition': None, 'candidates': ['agent'], 'returned': ['agent'], 'selected': ['agent']}`。
- **能回答什么问题**：为什么走了这条路？候选有哪些、路由器原样返回了什么、最后选中了谁？——这是第 7 节的主题。

### 3.9 `checkpoint_created`

- **何时发出**：两处。① `_commit` 里每次 checkpoint 落盘后（`runtime.py:491-498`），此时 `status` 通常是 `running`；② `_drive` 结尾，终止 checkpoint 的 status 被改写后**再发一次**（`runtime.py:276-288`）。
- **必填 keys**：`checkpoint_id`、`parent_id`、`next_nodes`、`status`。实际还带 `superstep`。
- **能回答什么问题**：DAG 有几个节点、父子关系如何、每个快照之后打算去哪、状态是什么？
- **副作用提示**：因为第 ② 处会重发，同一份 trace 里 `checkpoint_created` 的条数**可以比 checkpoint 数量多**。仓库自带 trace `traces/r_70b323a6.jsonl` 就是 6 条事件 / 5 个唯一 id（第 11 节第 5 条）。写消费方时必须按 `checkpoint_id` 去重、取最后一条为准。

### 3.10 `interrupt`

- **何时发出**：`_commit` 里发现本步产生了中断 payload 时（`runtime.py:482-490`），紧跟在该 checkpoint 的落盘之后。
- **必填 keys**：`checkpoint_id`、`node`、`payload`、`resume_schema`。
- **字段来源**：节点通过写保留 channel `__interrupt__` 触发，payload 形如：

  ```python
  {"node": "ask", "payload": {"q": "go?"}, "resume_schema": {...}, "resume_into": "apply"}
  ```

  `node` 缺省为 `"?"`，`resume_schema` 缺省为 `None`（`runtime.py:486-489`）。
- **能回答什么问题**：为什么停了？停在哪？在等人类给什么形状的数据？

### 3.11 `node_error`

- **何时发出**：节点函数抛异常时（`runtime.py:366-372`），捕获 `Exception`（不含 `BaseException`）后立刻发。
- **必填 keys**：`node`、`error_type`（异常类名）、`error`（`str(exc)`）、`traceback`（`traceback.format_exc()` 全文）。
- **能回答什么问题**：哪个节点以什么异常挂了、完整栈是什么？——节点异常是**数据不是崩溃**，这是刻意的设计：一次失败运行本身就是要观察的对象，不能因为一个节点炸了就把 trace 丢掉。

---

## 4. Pregel / BSP 式 superstep 执行模型

`runtime.py:6-16` 的模块注释就是整个模型：

```python
while active:
    plan    = dedupe(active)                # nodes scheduled this step
    writes  = run all plan nodes            # in parallel (threads)
    merge   writes into state via reducers  # -> state_merge event
    ckpt    = checkpoint(state, next)       # -> DAG grows by one node
    active  = next                          # -> route_decision events
```

主循环在 `_drive` 里（`runtime.py:224-268`），`invoke` 和 `resume` **共用同一份循环**——`_drive` 的 docstring 明说这是刻意的：「a second, near-identical loop is exactly how a debugger ends up lying about one code path.」（`runtime.py:205-211`）。

### 4.1 一个 superstep 里的五件事

```python
plan = list(dict.fromkeys(active))                      # 去重保序
self.emitter.emit(SUPERSTEP_START, index=superstep, nodes=plan)
before = state.clone()                                  # 合并前快照（用于 diff）
writes, failed = self._run_superstep(plan, cfg, state)   # 跑，不合并
next_active = self._merge_and_route(state, before, writes, plan)  # 合并 + 路由
parent = self._commit(...)                              # 落盘
```

### 4.2 为什么同一 superstep 的节点看到同一份 pre-merge 快照

`_run_superstep` 的第一行（`runtime.py:323`）：

```python
snapshot = state.to_dict()   # one frozen view for all parallel nodes
```

`State.to_dict()` 返回的是 `copy.deepcopy(self.values)`（`state.py:215-217`）。所以：

1. 所有节点拿到的是**同一份内容的独立副本**——它们彼此看不到对方的写入。
2. 节点也**不可能**改到 live state：它拿到的根本就是拷贝。这条规则是整个内核的地基，模块注释写得很直白：「A node never mutates state; it returns a patch. That single rule is what makes parallel fan-out safe and time travel possible.」（`runtime.py:15-16`）

合并只发生在 `_run_superstep` 返回之后，即所有节点都跑完之后。这就是 BSP 的 barrier。

### 4.3 并发如何限制

`max_concurrency` 只做一件事：决定线程池大小。

```python
max_workers = min(max(1, cfg.max_concurrency), len(plan))
with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lens-node") as pool:
    results = list(pool.map(run_one, plan))
```

（`runtime.py:317-322`）

细节：

- `RunConfig.max_concurrency` 默认 **4**（`runtime.py:82`）。
- `len(plan) == 1` 时**不走线程池**，直接调用（`runtime.py:328-329`）——单节点是最常见的情况，省一次线程创建。
- 实际 worker 数取 `min(max_concurrency, len(plan))`，所以 `max_concurrency=8` 但只有 2 个节点时就是 2 个 worker。
- `RunResult.max_concurrency` 记的是 `max(1, cfg.max_concurrency)`（`runtime.py:218`、`300`），即**配置值**，不是实际用到的 worker 数。
- 线程之间唯一共享的可变对象是 `Emitter`，它的 `emit` 用 `threading.Lock` 保护（`events.py:217-231`）。没有这把锁，两个节点可能拿到同一个 `seq`，trace 的总序就没了——而 `seq` 是所有消费方排序的依据（`events.py:195-200` 的 docstring 明确说了这一点）。
- 失败信息**不放在 `self` 上**：`_run_superstep` 的 docstring 说明，同一步两个节点可能在不同线程上失败，共享属性会丢掉其中一个。所以失败是返回值（`runtime.py:316-321`、`328-331`），多个失败用 `"; "` 拼起来。

### 4.4 为什么 reducer 是唯一合并点

因为节点拿不到 state，它只能返回 patch。patch 不经过 reducer 就无法进入 state：

- 节点返回 `dict` → `_run_superstep` 收集成 `(name, patch)` 列表
- `_merge_and_route` 逐个 `state.apply_patch(patch)` → 只有 `apply_patch` 会调 reducer（`state.py:240`）
- `apply_patch` 是 `State` 上唯一改动 `values` 的写入口

推论：**要改变合并语义，只有两个地方能动手**——在 schema 里给 channel 换 reducer，或者改 reducer 本身。没有第三条暗路。

### 4.5 循环的终止与兜底

三个出口：

1. `active` 为空 → 正常结束，`status = complete`。
2. 某步有节点失败 → `status = failed`，写一个 `next_nodes=[]` 的 failed checkpoint，`break`（`runtime.py:234-242`）。
3. 本步产生中断 → `status = interrupted`，写 interrupted checkpoint，`break`（`runtime.py:257-259`）。

外加一个兜底：`superstep` 达到 `max_supersteps`（默认 **100**，`runtime.py:55`）而 `active` 仍非空 → 抛 `GraphRecursionError`，被 `except` 捕获后转成 `status = failed`、`error = str(exc)`（`runtime.py:263-268`）。注释说了为什么要有它：「a learning tool should fail with a clear message, not hang.」手写图里忘了终止条件是常见错误，挂死比报错难查得多。

`supersteps` 字段的语义在正常结束时是「执行了几个 superstep」，在中断/失败时是「出事那一步的索引」（`break` 发生在 `superstep += 1` 之前）。实测：单步内中断的 run，`run_end` 里 `supersteps=0`，而 checkpoint 的 `superstep` 也是 `0`、`superstep_start.index=[0]`——即第 0 步被中断了，不是「一步都没跑」。见第 11 节第 6 条。

---

## 5. Reducer 语义

Reducer 签名是 `Callable[[Any, Any], Any]`：`(current, patch) -> merged`（`state.py:20-21`）。`current` 在 channel 为空时是哨兵 `MISSING`，不是 `None`——这个区别很重要，否则无法区分「没写过」和「写过 None」。

四个内置 reducer（`state.py:51-94`）：

### `overwrite`（默认）

```python
def overwrite(current: Any, patch: Any) -> Any:
    """Default reducer: last write wins. Simple, and lossy under fan-out."""
    return patch
```

`state.py:51-53`。docstring 自己就承认「lossy under fan-out」——这正是需要下面三个的原因。

### `add`

```python
if current is MISSING:
    return copy.deepcopy(patch)
if isinstance(current, (list, tuple)) and isinstance(patch, (list, tuple)):
    return list(current) + list(patch)
return current + patch
```

`state.py:56-62`。list/tuple 走拼接，其余走 `+`（数字相加、字符串相连都是它）。空 channel 时 `deepcopy(patch)` 而不是直接引用，避免调用方后来改了原对象污染 state。

### `add_unique`

`state.py:65-77`。先按 `add` 合并，再保序去重。去重键的选择：标量（`str`/`int`/`float`/`bool`/`None`）用自身，其它用 `repr(item)`。所以两个内容相同但非标量的对象**不会**被去重——这是 `repr` 的近似，不是深相等。返回类型是 `list`。

### `merge_dict`

`state.py:80-86`。空 channel 时 `dict(patch)`；否则 `dict(current)` 后 `update(patch)`。**浅合并**：嵌套 dict 的值会被整体替换。

### 声明方式：`Annotated`

reducer 通过注解声明，机制照搬 LangGraph：

```python
class AgentState(TypedDict):
    question: str
    messages: Annotated[list[dict[str, str]], add]
    tools_used: Annotated[list[str], add_unique]
    answer: str
    steps: int
```

（`examples/agent.py:27-35`）

`channels_from_annotations`（`state.py:150-174`）的解析规则：

- 名字以 `_` 开头 → **跳过**，不建 channel
- `get_origin(annotation) is Annotated` → `args[0]` 是 base 类型，`args[1:]` 里第一个「可调用且不是类」的元数据就是 reducer
- 其它情况 → `overwrite`

有个坑被专门写在 `resolve_annotations` 的 docstring 里（`state.py:128-147`）：如果定义模块用了 `from __future__ import annotations`（很常见），所有注解都变成**字符串**，`get_origin("Annotated[list[str], add]")` 返回 `None`，于是每个 channel 都静默退化成 `overwrite`。这个 bug 在单节点图上看不出来，只有 fan-out 开始丢写才暴露。解法是 `typing.get_type_hints(schema, include_extras=True)` 把字符串求值回来，并保留 `Annotated` 元数据；求值失败时降级用原始注解，而不是让一个类型注解把图定义搞崩。

### 重点：并行写入为什么必须逐个 `apply_patch`

这是本项目踩过的真实 bug，`_merge_and_route` 的 docstring 记录得很清楚（`runtime.py:392-399`）：

> Each write is applied **individually**. Collapsing them into one dict first (`merged.update(patch)`) looks equivalent and is not: with two parallel nodes both writing `messages`, the second *replaces* the first before the reducer ever runs, so one node's output silently vanishes. The reducer is the merge, so it must see every write.

看代码就明白差别在哪：

```python
# 只用于人读的摘要 —— 注意这行不会写入 state
flattened: dict[str, Any] = {}
for _node, patch in writes:
    if patch:
        flattened.update(patch)

# 真正写入：每个 patch 单独过一次 reducer
changed: list[str] = []
for _node, patch in writes:
    if patch:
        for name in state.apply_patch(patch):
            if name not in changed:
                changed.append(name)
```

（`runtime.py:405-410`）

假设两个并行节点都写 `messages`（reducer 是 `add`），一个写 `[a]`，一个写 `[b]`：

| 做法 | 结果 |
| --- | --- |
| 先 `merged.update()` 再一次性合并 | `{[a], [b]}` → update 后只剩 `[b]` → reducer 看到 `MISSING` 和 `[b]` → 最终 `[b]`，**`a` 丢了** |
| 逐个 `apply_patch`（当前实现） | 第一次 → `[a]`；第二次 → `add([a], [b])` → `[a, b]`，**两条都在** |

误差是**静默**的：没有异常、没有警告，trace 表面上一切正常，只是少了一条消息。测试 `test_parallel_writes_accumulate_not_overwrite`（`tests/test_kernel.py:106`）就是钉这个的。

注意 `flattened` 并没有被删掉——它服务于 `state_merge` 事件的 `patches` 字段（人读摘要），**不能**被误当成写入路径。代码里这行注释 `# for the human-facing summary only`（`runtime.py:403`）就是防线。

### 变更检测

`apply_patch` 只在「真的变了」时才记录：

```python
merged = channel.reducer(current, patch_value)
if merged is not current or current is MISSING or merged != current:
    self.values[name] = merged
    changed.append(name)
```

（`state.py:239-243`）

所以 `overwrite` 写入一个与当前值相等的内容，**不会**出现在 `changed_channels` 里。写 `messages` 这类每次返回新 list 的 reducer 则一定会出现。

### 保留 channel

`__interrupt__` 和 `__error__`（`state.py:45`）不需要在 schema 里声明，`apply_patch` 遇到它们会直接赋值（`state.py:230-234`）。理由写在 `graph.py:218-222`：checkpoint 把这两个 channel 一起存了，恢复暂停的 run 时如果拒收它们，每次 resume 都会被自己的存档卡住。

`summarize_state` 会**隐藏**所有 `__` 开头的键（`events.py:276-277`），所以中断 payload 不会漏进普通状态摘要里。

---

## 6. Checkpoint DAG

模块注释（`checkpoint.py:1-10`）一句话说清了核心：每个 checkpoint 都记 `parent_id`。线性运行看着像条链，但从旧 checkpoint resume 的那一刻它就变成真正的 DAG——一个父节点两个孩子，两条不同的未来。

### 6.1 `checkpoint_id` 格式

```python
checkpoint_id=f"{thread_id}_{superstep:04d}_{uuid.uuid4().hex[:6]}"
```

（`checkpoint.py:121`）

三段：`thread_id` + 4 位零填充的 superstep + 6 位随机十六进制。示例（仓库自带 trace 实测）：`demo_0000_3f7364`、`demo_0003_acd875`。

**注意**：任务描述里写的是 `{uuid6}`，代码实际用的是 `uuid.uuid4().hex[:6]`——是 uuid4 的前 6 个 hex 字符，不是 UUIDv6。写消费方时不要把它当 UUID 解析（第 11 节第 1 条）。

这个 id 同时是文件名：`CheckpointStore._path` 拼 `<dir>/<checkpoint_id>.json`（`checkpoint.py:95-96`）。所以 `thread_id` 里不能有路径分隔符。

### 6.2 `parent_id` 如何形成 DAG

每次 `_commit` 都把当前 parent 的 id 写进新 checkpoint（`runtime.py:478`）：

```python
parent_id=parent.checkpoint_id if parent else None
```

- 第一个 checkpoint 的 `parent_id` 是 `None` → 它是根。
- 正常推进时 parent 就是上一步的 checkpoint → 链。
- **从旧 checkpoint resume** 时，`parent` 是那个旧 checkpoint → 新分支挂上去，DAG 分叉。

查询接口：

- `Checkpoint.lineage(store)` —— 沿 `parent_id` 一路走到根，返回**从旧到新**的链（`checkpoint.py:66-79`）。带环保护（`seen` 集合），存档损坏也不会死循环。
- `Checkpoint.children(store)` —— 找 `parent_id` 等于自己的所有 checkpoint，即「从这儿分叉出去的实验」（`checkpoint.py:81-83`）。
- `CheckpointStore.dag(thread_id=None)` —— 直接吐 `{"nodes": [...], "edges": [{"source": parent, "target": self}]}`，给渲染器用（`checkpoint.py:164-182`）。

`render_checkpoint_dag` 只用 `checkpoint_created` 事件就能画出这棵树（`render.py:265-292`），不需要碰 checkpoint 文件。

### 6.3 状态机

```python
STATUS_RUNNING     = "running"      # 刚落盘，还不知道怎么结束
STATUS_COMPLETE    = "complete"     # 正常走到 active 为空
STATUS_INTERRUPTED = "interrupted"  # 等人：一等公民状态，不是错误
STATUS_FAILED      = "failed"       # 节点异常，或超过 max_supersteps
STATUS_RESUMED     = "resumed"      # 已定义并导出，但当前代码没有地方写入
```

（`checkpoint.py:24-30`）

`interrupted` 的定位在注释里写得很清楚：「``interrupted`` is a first-class state, not an error — it is the anchor a human-in-the-loop resume starts from.」（`checkpoint.py:26-30`）

状态流转有三条路径：

1. `_commit` 时 `status=status or "running"` —— 没特别说明就是 running（`runtime.py:479`）。
2. 中断时直接建成 `interrupted`，并挂上 `interrupt` 字典（`runtime.py:249-254`）。
3. 失败时直接建成 `failed`，`next_nodes=[]`（`runtime.py:237-241`）。
4. **收尾改写**：`_drive` 结尾，如果终止 checkpoint 的 status 与本次 run 的最终 status 不同，就改写并重新落盘（`runtime.py:276-288`）。这是「complete」这个状态唯一的来路——中间 checkpoint 全是 running，只有最后那个在收尾时被改写。

`STATUS_RESUMED` 是个例外：它被定义、被 `__init__.py` 导出，但 `grep` 全仓只有定义和导出两处，没有任何代码写入它（第 11 节第 7 条）。

### 6.4 原子写入

```python
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(checkpoint.to_dict(), fh, ensure_ascii=False, indent=2, default=_json_default)
os.replace(tmp, path)  # atomic: a truncated checkpoint can never be read
```

（`checkpoint.py:102-105`）

写临时文件再 `os.replace` —— 同目录内的 `replace` 在 POSIX 和 Windows 上都是原子替换（不跨卷）。所以一个被读到一半的 checkpoint 文件不可能存在：读者要么看到旧的完整版本，要么看到新的完整版本。这对 `watch` 模式和并发读者是硬需求。

`_json_default`（`events.py:113-121`）负责把非 JSON 类型降级：有 `model_dump()` 用 `model_dump()`，有 `to_dict()` 用 `to_dict()`，set 排序后输出，其余 `repr()`。所以 checkpoint 里塞任何对象都不会让写入崩，只会被 repr 化。

### 6.5 存储布局与读取

一个 checkpoint 一个文件，扁平放在 `<dir>/` 下：`<dir>/demo_0000_3f7364.json`。

- `put()` 写完顺手更新内存缓存 `_cache`（`checkpoint.py:106`）。
- `load()` 先查缓存，再查文件，命中后回填缓存（`checkpoint.py:133-142`）。
- `all(thread_id=None)` **每次重扫目录**并重新 `json.load`，跳过解析失败的文件（`except (json.JSONDecodeError, TypeError): continue`），最后按 `(ts, superstep)` 排序（`checkpoint.py:144-157`）。这是「故意笨」的部分：慢，但任何时候手改一个 JSON 都能被读到。
- `latest(thread_id)` 取 `all()` 的最后一个（`checkpoint.py:159-161`）。

---

## 7. 路由透明性

大多数框架里，路由器返回一个字符串，图就跳走了。你能看到「跳到了哪」，看不到「本来能跳到哪」，更看不到「路由器原样返回了什么」——而后者恰恰是「字符串和节点名恰好同名」这类 bug 藏身的地方。

lens-harness 的选择是：**一次路由决策，记三个层次的信息**。

### 7.1 `route_decision` 的五个字段

```python
{
    "source": node,               # 谁在路由
    "condition": ce.router_name,  # 哪个路由器（静态边时为 None）
    "candidates": ce.candidates,  # 全部候选键
    "returned": returned,         # 路由器原始返回值，未做任何解释
    "selected": targets,          # 最终要跑的节点列表
}
```

（`graph.py:254-261`）

- `candidates` 来自 `ConditionalEdge.candidates`，就是 `list(mapping)`（`graph.py:71-73`）——**声明时给的 map 的键全集**，不是「可能返回的值」的推测。
- `returned` 保留原始类型：路由器返回 `str` 就是 `str`，返回 list 就是 list。实测条件边形态里 `returned` 是 `'tools'`（字符串），静态边形态里是 `['agent']`（列表）。
- `selected` 是 `[mapping[key] for key in keys]` 映射**之后**的节点名（`graph.py:253`）。

这三个的差别正是重点：`returned='tools'` 与 `selected=['tools']` 在键和值同名时看起来一样，不同名时就是救命的对照。

### 7.2 `mapping` 是 `return_value -> node`

`add_conditional_edges`（`graph.py:114-140`）的 docstring 写得很直接：

> ``mapping`` is ``router return value -> target node``, matching LangGraph. Descriptions are a *separate* channel (``descriptions``) because overloading the mapping values with prose is how a graph ends up routing to a node literally named "the agent requested a tool call".

范例里就是标准的写法（`examples/agent.py:137-149`）：

```python
graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        END: END,
    },
    name="route_after_agent",
    descriptions={
        "tools": "the agent requested a tool call",
        END: "the agent produced a final answer",
    },
)
```

这里 `"tools": "tools"` 的左边是路由器的返回值，右边是节点名。如果贪方便写成 `{"tools": "the agent requested a tool call"}`，图就会去找一个叫「the agent requested a tool call」的节点——所以人读说明走独立参数。

### 7.3 为什么 `candidates` 是全集

因为「候选有哪些」和「选中了谁」是两个不同的问题，而调试时两个都要。`candidates` 是**声明时**的 map 键全集，所以它不依赖任何运行时输入——即使路由器这次只可能返回其中一个，trace 里也记着另外几个没被选中的可能。

渲染器把这个意图做成了视觉中心（`render.py:207-224`）：

```
⑂ route after agent via route_after_agent()
    candidates: tools | __end__
    returned:   'tools'
    selected:   tools
```

选中的高亮加粗，未选中的变暗。静态边则简化成一行箭头：`→ edge tools → agent`。

### 7.4 无效返回值的处理

路由器返回一个不在 `mapping` 里的键，`next_targets` 直接抛 `GraphError`（`graph.py:247-252`）：

```
router 'route_after_agent' returned 'toolz', which is not a declared candidate ['tools', '__end__']
```

这是「宁可炸也不猜」：静默地走到一个不存在的节点，或者悄悄回退到默认边，都会产生一份看起来正常但路由错误的 trace。测试 `test_conditional_routing_records_candidates_and_choice`（`tests/test_kernel.py:156`）覆盖了正常路径。

### 7.5 路由器拿到的状态是只读副本

```python
read_only = state.to_dict()   # deepcopy
returned = ce.router(read_only)
```

（`graph.py:241-242`）

路由器不能靠修改入参来影响图。它只能通过返回值表达意见。

---

## 8. 时间旅行

`Runtime.resume()`（`runtime.py:129-192`）是「从任意 checkpoint 继续，且不重跑已经执行过的节点」。

### 8.1 为什么不会重跑

`invoke` 从 `active = [entry]` 开始（`runtime.py:119`），`resume` 从 checkpoint 里存的 `active` 开始：

```python
active = list(checkpoint.next_nodes) or [resume_node] if resume_node else list(
    checkpoint.next_nodes
)
```

（`runtime.py:181-183`）

`next_nodes` 是那个 checkpoint 落盘时**已经算好、还没执行**的节点列表。所以 resume 直接从「中断点之后的下一步」进入循环，前面的节点一次都不执行。测试 `test_resume_replays_from_old_checkpoint_without_rerunning_past_nodes`（`tests/test_kernel.py:257`）用计数器验证了这点。

`_drive` 的起始 `superstep` 也是续上的：`checkpoint.superstep + 1`（`runtime.py:190`），所以新事件里的 `superstep` 不会从 0 重来。

### 8.2 三个必须处理的坑

**(1) 清除 `__interrupt__`**

```python
state.values.pop("__interrupt__", None)
```

（`runtime.py:154`）

中断信号住在被 checkpoint 的状态里。不清掉的话，恢复后的下一步会立刻再次触发中断，永远停在刚恢复进来的那个节点后面。`runtime.py:148-153` 的注释记下了症状：「resume "works", runs one node, then reports interrupted again.」payload 本身已经存在 `checkpoint.interrupt` 里，所以清这个 channel 不丢任何信息。

**(2) payload 是 channel patch，不是 node 入参**

```python
changed = state.apply_patch(payload)
```

（`runtime.py:166-168`）

`resume_into` 指的是「恢复后从哪个节点进入」，**不是** channel 名。如果把 payload 写到节点名下（例如 `{"apply": ...}`），`apply_patch` 会报 `unknown channel 'apply'`。同时 payload 必须是 dict，否则抛 `GraphError`（`runtime.py:162-168`）。

**(3) payload 本身也是一次状态写入，要进 trace**

```python
self.emitter.emit(
    STATE_MERGE,
    patches=summarize_state(payload),
    changed_channels=changed,
    state=summarize_state(state.values),
    origin="resume_payload",
)
```

（`runtime.py:173-179`）

这条 `state_merge` 的 `origin` 是 `"resume_payload"`。渲染器据此把标签从 `state` 改成 `state · resume payload`（`render.py:191-199`）。这样 trace 里能分清「图写的」和「人写的」——否则人类的输入会在回放中凭空出现。

### 8.3 从任意 checkpoint 继续 = 分叉实验

`resume` 的 docstring（`runtime.py:136-141`）把它定义成 fork 原语：

> This is also the fork-experiment primitive: resume an *old* checkpoint with a patched payload and the DAG grows a second branch, because the new checkpoint's `parent_id` points at the old one.

流程是：拿一个旧 checkpoint id → `resume(ckpt, payload={...})` → 新 checkpoint 的 `parent_id` 指向旧的那个 → 用 `Checkpoint.children()` 就能看到「同一个起点，两个未来」。

这里有一个限制：**带 payload 的 resume 必须落在有中断的 checkpoint 上**。若目标 checkpoint 没有 `interrupt` 信息，`resume_node` 是 `None`，带 payload 会抛 `GraphError`（消息里含 `interrupt`、`resume into` 字样，`runtime.py:157-161`）。不带 payload 的 resume 则可以从任何 checkpoint 走——那是最纯粹的「回到过去再走一遍」。

### 8.4 resume 的边界（如实列出）

- **不发 `run_start`**：resume 的事件从 `superstep_start` 开始，所以一段独立的 resumed trace 里没有图快照。
- **不新建 `run_id`**：`_drive` 里 `self.emitter.run_id = checkpoint.run_id`（`runtime.py:212`），同一个 run 的续篇沿用原 id。
- **事件 `thread_id` 从 parent 继承，checkpoint 前缀跟着它走**：`self.emitter.thread_id = parent.thread_id if parent is not None else cfg.thread_id`（`runtime.py:213`），而 `_commit` 里 `thread_id=self.emitter.thread_id`（`runtime.py:474`）。也就是说 resume 时传进来的 `config.thread_id` **不会**改变 thread —— 新 checkpoint 的前缀仍是 parent 的 thread。两者早先会分叉，现已统一（第 11 节第 2 条记录了这次修复）。
- **不校验图是否变了**：resume 只用当前内存里的 `compiled` 图执行 checkpoint 里存的 `next_nodes`。如果图定义改了，节点名对不上会直接 `KeyError`（`runtime.py:347` 取 `graph.nodes[name]`），没有任何友好检查。

---

## 9. 与 LangGraph 的对照

本项目是**学习用途**，对照是为了说明「哪些抄了、哪些简化了、哪些没有」，不是竞争关系。

| 维度 | LangGraph | lens-harness | 备注 |
| --- | --- | --- | --- |
| 状态 schema | `TypedDict` + `Annotated[T, reducer]` | 同 | `state.py:150-174`，机制照搬，30 行实现 |
| 内置 reducer | `add_messages`、`operator.add` 等 | `add` / `add_unique` / `merge_dict` / `overwrite` | 按「消息/集合/字典/覆盖」四种最常见的形态收口 |
| 节点契约 | 返回 partial state | **完全相同**：`Callable[[dict, NodeContext], dict \| None]` | `graph.py:36` 附近 |
| 执行模型 | Pregel / BSP superstep | 同 | `runtime.py:6-16` |
| 并发 | 可配置，生产级调度 | `ThreadPoolExecutor`，`max_concurrency` 默认 4 | 无线程池复用、无超时、无重试 |
| 路由 | `add_conditional_edges(source, router, mapping)` | 同签名 + 额外 `descriptions` 参数 | `graph.py:114-140` |
| 路由可观测性 | 需自行埋点 | `route_decision` 事件记全三层信息 | 见第 7 节 |
| checkpoint | `checkpointer` 抽象，多种后端（内存/SQLite/Postgres） | 单文件 JSON + 原子写 | 没有后端的后端，可 `cat` |
| 时间旅行 | 支持，需要 checkpointer | 支持，`resume(checkpoint_id, payload=...)` | 见第 8 节 |
| 中断 / human-in-the-loop | `interrupt()` + `Command(resume=...)` | 节点写 `__interrupt__` channel + `resume(payload=...)` | 机制不同但语义对应 |
| 流式输出 | `stream()` / `astream()` 多种模式 | `ctx.emit_token()` → `node_token` 事件 | 只有 token 级，没有 state 级流 |
| 事件 / 追踪 | 依赖 LangSmith 或自建回调 | `Event` + `EVENT_SCHEMA` 内置 | 这是本项目的**主产物** |
| 依赖 | 一堆 | **零**（纯 stdlib） | `llm.py:156-161` 连 HTTP 都用 `urllib` |
| 分布式 / 持久执行 | 有相关支持 | **没有** | 见第 10 节 |

一句话总结差异：LangGraph 把 trace 当成可选的观测设施，lens-harness 把 trace 当成**唯一产物**，图只是产生 trace 的手段。

---

## 10. 明确不做的清单

以下都是**决定不做**，不是「还没做」。判断标准仍是那句：不帮你看清事情的功能不做。

### 不做分布式与持久执行

没有消息队列、没有 worker 池、没有跨进程调度、没有崩溃恢复。`Runtime` 是一个进程内的 `while` 循环。理由是：分布式会让「一步里发生了什么」这个问题变得需要跨机器拼装答案，而这个项目的全部价值就在于答案能在一屏里读完。

### 不做生产级存储后端

只有 `CheckpointStore` 一种实现：目录 + 一个 checkpoint 一个 JSON。不接 SQLite、不接 Postgres、不做压缩、不做 GC。旧的 checkpoint 会一直堆着，删不删是你的事。

### 不做认证、多租户、配额

没有用户概念。`thread_id` 只是一个字符串前缀，不是安全边界。

### 不做通用 Web 前端（早期版本写的是「不做 Web UI」）

这一条**曾经是「不做 Web UI」，M2 之后作废**——第 12 节讲的浏览器 lens 已经存在。作废的范围需要说清楚，否则它会退化成一句空话：

**做了的**：一个只读页面，看一份已有的 trace。没有构建步骤、没有 npm、没有框架、没有路由、没有 CDN 资源，单 HTML 文件 + 一个标准库 `http.server`。

**仍然不做的**：

- ~~**不做写操作**~~（Studio 之后收窄，见下节）主 lens 页面仍然是只读的：不能运行图、不能 resume、不能改 checkpoint、不能提交中断的审批。它只有一个方向——trace → 屏幕。
- **不做通用 Web 应用**：没有账号、没有会话、没有多用户、没有 WebSocket（要实时就用浏览器原生的 `EventSource`）。
- **不做前端工程化**：不引 React/Vue/Svelte，不引打包器，不引 TypeScript，不引测试框架。理由是原则二——如果画一个 trace 需要先装 300MB 依赖，那这个项目的可读性就没了。

### Studio 的写路径边界（Studio 之后新增）

`/studio` 是一个**新的写入口**，与只读 lens 并存，边界如下：

- **做了**：一个单用户、单会话的编排工作台——左侧对话流（结构化命令：初始化模板、增删改节点/边、设置入口、运行），后端把命令落到 JSON spec 上、每次运行都走**普通内核路径**（build_graph → Runtime → trace/checkpoints），右侧用 iframe 复用只读 lens 看生成的 trace。命令错误（未知通道、环路、无法编译的 spec）原样回显到对话流，不静默降级。
- **仍然不做**：自然语言理解——聊天是结构化命令，不是 chatbot；把「改节点」翻译成自然语言是另一个项目的事。对话消息不进 LLM。
- **仍然不做**：多用户/持久化会话——Studio 会话在进程内存里，服务重启即清空；spec 的持久形态仍是 JSON 文件（可从对话流复制导出）。
- **仍然不做**：绕过内核——Studio 没有自己的执行器，它只是 `run_spec` 的一个远程面板；trace、checkpoint、diff 与 CLI 完全同构。
- 依然没有认证：Studio 只应绑定在回环地址或可信网络里运行。
### 不做任意 trace 的合并视图（M3 之后收窄）

这条**曾被写成「不做多 trace」，M3 之后需要重新划界**——`lens diff` 与 `serve a.jsonl b.jsonl` 已经存在（第 13 节）。边界现在是：

- **做了**：两份 trace 的**归一化行为对比**——先各自归一化（去掉每次运行必然不同的 5 个字段），再按结构键对齐。输出是 `same / changed / only-a / only-b` 四类，两端各有一个视图（终端 `diff` 子命令、浏览器 `/compare` 页）。
- **仍然不做**：把两份 trace **合并成一个执行视图**（时间轴、图画布混画两个 run）。对比页是分节罗列差异，不是把两个 run 叠在一张图上——叠画需要回答「哪个 run 的当前状态是『当前』」，那是一套新的状态语义，M3 不碰。
- **仍然不做**：两个以上的 trace。对比是两两关系；N 份 trace 的「聚类」是另一个工具。
- **仍然不做**：多 trace 的实时 follow 对比。两份都在写的 trace 无法定义「对齐到哪一刻」。

### 编排入口的边界（DSL 之后收窄）

这条**曾隐含「编排只能写 Python」**。JSON DSL（`dsl.py` + `run --graph`）落地后重新划界：

- **做了**：一个**声明式编排入口**——JSON spec 编译成普通的 `StateGraph`，节点模板七种（`llm` / `tool_loop` / `interrupt` / `python`，加上声明式的 `transform` / `map` / `subgraph`），路由条件是内置两种（`tool_call` / `always`）或 `python:module:func` 引用。编译后的图与手写的图**不可区分**：同一内核、同一事件契约、同一 trace/diff 路径。spec 错误一律 `SpecError` 带 JSON 路径，不静默降级；节点运行期的 spec 违规（如 transform 引用不存在的通道、map 收到非列表）走内核的 `node_error` → run `failed`，同样响亮。`subgraph` 的子图有自己的 run/trace/checkpoint，父图通过 `run_into` 记录子 run_id——不合并两条 trace，只留关联键。
- **仍然不做**：可视化拖拽编排器（浏览器仍是只读 lens，见上）。
- **仍然不做**：插件/注册表式的节点类型扩展——七种写死了；加新类型要改 `dsl.py`，这属于内核演进而非编排行为。
- **仍然不做**：YAML 或其它格式——保持零依赖（原则一）。JSON 够表达图结构；`pyyaml` 的依赖代价买不来新能力。
- **仍然不做**：在 spec 里内嵌任意代码——`python` 节点与条件必须是可导入的 `module:func` 引用，代码在仓库里、可 review、可测。内嵌代码字符串会让「编排文件」变成「藏起来的程序」，trace 也无法对它负责。

### 不做流式的 state 输出

只有 `node_token` 这一种流（token 级）。没有「每一步的状态增量推给客户端」的通道——要看状态就读 `state_merge` 事件。

**M2 之后这条仍然成立，但需要一句澄清**：浏览器 lens 是跟着事件流推进的，每来一条 `state_merge` 它就重画——但服务端推的仍然是**完整事件**，不是「状态差分补丁」。页面的状态永远是「重放事件前缀」的结果（原则二），从来不是「把补丁叠加到上一帧」。所以这里没有增量状态协议，也没有它的正确性问题。

### 不做 async / await

内核是同步的，并发靠线程（`runtime.py:332`）。没有 `asyncio` 事件循环、没有 `astream`。

### 不做完整的事件 schema 校验

`validate_event` 只检查**必填键是否存在**，不检查类型、不拒绝多余键（`events.py:60-68`）。所以 `state_merge` 能带 `diff` 和 `origin`，`run_end` 能带 `checkpoint_id` 和 `error`，而校验器不管。这是刻意的宽松：契约要保证「消费方需要的键一定在」，而不是「不许有别的键」。

### 不做权限/沙箱

节点函数能读环境变量、能发网络请求、能写文件。`examples/agent.py` 里的 `_calculator` 用 `eval`（虽然前面有正则白名单，`examples/agent.py:41-48`）——它自己都标了 `noqa: S307`。这是个教学项目，不负责防住恶意节点。

### 不做自动重试

节点抛异常 → `node_error` → run 变 `failed`，结束。不重试、不退避、不降级。

### 不做多 LLM provider 抽象

只有 OpenAI 兼容协议一种真实后端（`/chat/completions`，`llm.py:200-209`）+ `MockLLM`。加新协议要自己写一个满足 `LLM` Protocol 的类（`llm.py:41-44`），没有注册表、没有插件系统。

---

## 11. 已知落差与不一致（自查记录）

以下每一条都是写文档时对着代码和实际运行核出来的。**如实列出，不替代码圆场。**

需要注意时间线：这份文档写作期间，仓库被持续修改。下面有些条目在写作时成立、**现在已经修掉**——这样的条目保留下来并标上「（已修复）」，因为它们记录了「曾经哪里会骗人」，也是这个项目自己的口径说明。**标了「（已修复）」的，正文断言以修复后的代码为准；没标的，现在是问题。**

### 11.1 `checkpoint_id` 用的是 uuid4 前 6 位，不是 UUIDv6

`checkpoint.py:121` 实际是 `f"{thread_id}_{superstep:04d}_{uuid.uuid4().hex[:6]}"`。任务描述里写的 `{uuid6}` 与代码不符。**后缀是 6 个 hex 字符**（24 bit），碰撞概率在长跑里并非可忽略，但代码没有任何去重或重试逻辑——`put()` 直接覆盖同名文件。

### 11.2 resume 时事件 `thread_id` 与 checkpoint id 前缀会分叉（已修复）

写作时的行为：

- checkpoint id 前缀：`cfg.thread_id`（`_commit` 里，旧代码）
- 事件 `thread_id`：`parent.thread_id if parent is not None else cfg.thread_id`（`runtime.py:213`）

当时实测：`invoke(thread_id="alpha")` 之后 `resume(ckpt0, config=RunConfig(thread_id="BETA"))`，新 checkpoint 叫 `BETA_0001_66c1c7`，但 `RunResult.thread_id` 和所有新事件的 `thread_id` 仍是 `"alpha"`。同一份 trace 里，checkpoint 的 thread 和事件的 thread 对不上——一段 trace 会同时声称自己属于两个 thread。

**修复**：`_commit` 现在写 `thread_id=self.emitter.thread_id`（`runtime.py:474`），即 checkpoint 前缀跟随事件。重新实测同一个实验：新 checkpoint 变成 `alpha_0001_32096d`，`ckpt.thread_id`、`RunResult.thread_id`、全部新事件的 `thread_id` 都是 `"alpha"`，一致。

**遗留语义（不是 bug，但要知道）**：`resume(config=RunConfig(thread_id="BETA"))` 里的 `thread_id` 现在**完全不生效** —— resume 的 thread 永远来自 parent checkpoint。想换 thread 只能重新 `invoke`。

### 11.3 `resume()` 不发 `run_start`（已修复）

写作时的实测事件序列从 `superstep_start` 开始：`superstep_start, node_start, node_end, state_merge, route_decision, checkpoint_created, checkpoint_created, run_end`。没有任何 `run_start`。后果：一段独立的 resumed trace 里**没有图快照**，`render_events` 会缺掉 `▶ run ...` 那一段，lens 无法从这份 trace 单独画出静态图。

**修复**：`resume()` 现在在进入循环前发一条 `RUN_START`，带上与 `invoke` 相同的 `graph` 快照和 `inputs`（来自 checkpoint 状态的摘要），外加 `resumed_from=<checkpoint_id>` 标记。终端渲染器在 `▶ run ...` 行尾标注 `(resumed)`；浏览器 lens 记录 `vm.resumedFrom`。回归测试：`test_resume_emits_run_start_so_a_resumed_trace_is_self_describing`（`tests/test_kernel.py`）。

### 11.4 `replay_state_at` 重建的是「摘要」，不是完整状态（已修复：改名 + CLI 优先读 checkpoint）

写作时 `render.py:308-332` 的实现取的是 `state_merge` 事件里的 `state` 字段，而那个字段经 `summarize_state` 截断过（`events.py:268-284`）。

实测：状态里 `big` 是 500 字符、`items` 是 40 项，`replay_state_at` 给出 `big` 长度 **132**（= 120 + `"…(500 chars)"`）、`items` 长度 **9**（= 8 项 + `"…(+32 more)"`），而 `checkpoint.state` 里是完整的 500 / 40。

**修复**：函数改名为 `replay_state_summary_at`——docstring 不再声称超出摘要的保真度。CLI 的 `state` 子命令现在**优先读取 trace 旁边 `checkpoints/<id>.json` 里的完整状态**，读不到才回退到 trace 摘要，并且输出第一行注明数据来源（`# full state from checkpoint file` / `# full state from trace summary (long values truncated)`）。回归测试：`test_replay_state_summary_is_truncated_but_checkpoint_is_full`。

### 11.5 同一个 checkpoint id 会发两条 `checkpoint_created`

`_commit` 发一次（`runtime.py:491-498`），收尾改写状态后再发一次（`runtime.py:276-288`）。所以 `checkpoint_created` 的条数可以比 checkpoint 数量多。仓库自带 trace 实测：6 条事件 / 5 个唯一 id，最后那个 id 先以 `running` 出现、再以 `complete` 出现。

副作用（**已被修掉，但契约本身没变**）：`render_checkpoint_dag` 用 `nodes = {e.data["checkpoint_id"]: e.data for e in events if e.kind == CHECKPOINT_CREATED}`（`render.py:268`）靠 dict 覆盖取到最后一条，这没问题；但 `list_checkpoints()` 早期版本是直接 `[e.data for e in events if e.kind == CHECKPOINT_CREATED]`，会把同一个 id 列两次，导致 `lens checkpoints` 打重复行。现在的实现（`render.py:335-352`）已按 id 去重、保留最后一条并保持首次出现顺序，docstring 里也写明了这个原因。**结论**：`checkpoint_created` 仍然是一个 id 可能出现多条的事件，消费方必须自己去重——只是仓库自带的那个消费方现在做了这件事。

### 11.6 `supersteps` 在中断/失败时不是「执行了几步」

`break` 发生在 `superstep += 1` 之前（`runtime.py:257-262`），所以中断时 `run_end.supersteps` 是**出事那一步的索引**。实测单步内中断：`status=interrupted`、`supersteps=0`、checkpoint `superstep=0`、`superstep_start.index=[0]`——明明跑了一个 superstep，计数器却是 0。正常结束的 5 步 run 则报 `supersteps=5`。同一个字段两种语义，消费方需要配合 `status` 才能解释。

### 11.7 `STATUS_RESUMED` 定义了但从未使用（已修复：删除）

写作时 `checkpoint.py:29` 定义、`__init__.py:19` 与 `:60` 导出，全仓 `grep` 找不到任何写入点。resume 之后的 checkpoint 状态是 `running`/`complete`，不是 `resumed`。

**修复**：选择**删除**而不是补写入。补写入需要给 resume 的锚点 checkpoint 改状态，那会覆盖 `interrupted`——而 §6.3 明确说 `interrupted` 是一等状态、是 human-in-the-loop resume 的锚点，改写它就破坏了这个契约。删除后 `STATUS_RESUMED` 不再出现在 `checkpoint.py` 与 `__all__`。回归测试：`test_status_resumed_is_gone`。

### 11.8 `run_start` 的图快照里 `descriptions` 装的是 mapping（已修复）

写作时的 `snapshot()` 把 `"descriptions"` 赋成了 `ce.mapping`，于是：

- `run_start.graph.conditional_edges[0].descriptions` = `{'tools': 'tools', '__end__': '__end__'}`
- `route_decision.descriptions` = `{'tools': 'the agent requested a tool call', '__end__': 'the agent produced a final answer'}`

同一张图的两个事件互相矛盾：静态快照里的「人读说明」被节点名覆盖了（`tools → "tools"` 这种同义反复）。演示图里恰好 `mapping` 的键值不同才看得出差别；如果路由返回值和目标节点同名，这个 bug 会完全隐形。

**修复**：`graph.py:193` 现在是 `"descriptions": dict(ce.descriptions),`，并且额外多给一个 `"mapping": dict(ce.mapping),`（`graph.py:194`）—— 快照同时暴露「说明文字」和「返回值 → 节点」的映射，两者不再混为一谈。重新实测：`snapshot` 的 `descriptions` = `{'x': 'the x branch'}`、`mapping` = `{'x': 'b'}`，与 `route_decision.descriptions` 相同（EQUAL: True）。

注意 `snapshot()` 现在把两者的语义分开了，但 `candidates` 字段仍然只是 `list(ce.mapping)`（`graph.py:71-73`），即「返回值全集」，不含说明。

### 11.9 `__error__` 保留 channel 没有任何消费者（已修复）

写作时 `state.py:45` 声明 `RESERVED_CHANNELS = ("__interrupt__", "__error__")`，注释说它用来「record a structured, non-fatal problem for the lens」。`apply_patch` 会接受节点写入（`state.py:230-234`），`graph.py:222` 也放行它进入初始状态。但 `runtime.py` **从不读它**：没有对应事件、没有 checkpoint 字段。

**修复**：新增第 12 种事件 `error_reported`（`events.py` 的 `EVENT_SCHEMA`：必填键 `node`、`problem`）。superstep 合并后，runtime 检查哪些节点在本步写入了 `__error__`，每个写入者发一条 `error_reported`。两个关键语义：(1) 只有**本步写入**的节点发事件——channel 会随 checkpoint 持久化，resume 时不得重报旧问题；(2) 它是非致命报告，run 照常继续，与 `node_error`（节点抛异常、run 转 `failed`）互补。终端渲染器画 `⚠ reported <node>: ...`；浏览器 lens 推进 `vm.errors` 但不改 `vm.status`。回归测试：`test_error_reported_event_gives_the_reserved_channel_a_consumer`（含 resume 不重报的断言）。

更进一层：`state_merge` 的 `diff` 现在显式过滤掉所有 `__` 开头的 channel（`runtime.py:421-428`，注释解释了为什么——否则会渲染出一行 `__interrupt__: null → null` 的无意义 diff），`summarize_state` 也一直隐藏它们（`events.py:276-277`）。所以写 `__error__` 的唯一效果是它躺在 checkpoint 的 `state` 里，任何事件、任何渲染器都看不到。`__interrupt__` 至少还有 `interrupt` 事件把它抬出来；`__error__` 连这个都没有。

### 11.10 `CheckpointStore` 的注释与实现不符（已修复）

写作时：类 docstring 说 `File-backed checkpoint store: <dir>/<thread>/ckpt_*.json + index.`，模块 docstring 说 `one JSON file per checkpoint plus an index file`；而实现是扁平 `<dir>/<checkpoint_id>.json` —— 没有 `ckpt_` 前缀、没有 thread 子目录、没有 index 文件。`_path` 里甚至算了一个 `thread_dir` 变量然后完全没用。注释比代码旧。

**修复**：模块 docstring（`checkpoint.py:8-9`）现在是 `Storage is deliberately dumb: one flat JSON file per checkpoint, <dir>/<checkpoint_id>.json, with no index — all() just scans the directory.`，类 docstring（`checkpoint.py:88`）是 `File-backed checkpoint store: flat <dir>/<checkpoint_id>.json files.`，`_path`（`:95`）直接 `return os.path.join(self.directory, f"{checkpoint_id}.json")`，无 `thread_dir` 变量。

这条曾经值得记下来的理由不是拼写：**「有 index」这个假象会让人以为 `all()` 是读索引、以为删除文件后索引会残留**。现在注释和实现都是「扫目录、无索引」，那 `all()` 每次调用 O(n) 扫盘（`checkpoint.py:144-157`，还跳过解析失败的文件）就是**明示的设计**，不是意外。

### 11.11 一个 `Runtime` 实例只能跑一次 run（已修复）

写作时 `Emitter` 在 `Runtime.__init__` 里创建一次（`runtime.py:99`），`invoke` 直接复用 `self.emitter.run_id`（`runtime.py:110`），`seq` 永不清零。实测（一个最小探针图，两次 `invoke()`）：两次拿到**相同**的 `run_id`，`seq` 接着上一轮 continue（共 18 条事件，两轮混在一条序列里），只有 checkpoint id 因 uuid 后缀不同而不同。要跑多次 run，就 new 一个 `Runtime` 实例。

**修复**：`invoke()` 现在每次生成新的 `run_id`、调用 `Emitter.reset_seq()`（锁内归零）、重置 `superstep=-1`。同一个 Runtime 可以连续 `invoke` 多次，每次是独立的一条 trace 序列。`resume` 不受影响——它沿用 checkpoint 的 `run_id`，这是刻意的（同一 run 的续篇）。回归测试：`test_one_runtime_can_invoke_twice_with_independent_runs`。

后果：想跑两次对比，必须每次新建 `Runtime`（`examples/agent.py:184` 就是这么做的，一次 run 一个实例）。docstring/文档里没有写这条约束。

### 11.12 `RecordingLLM` 缓存命中被标记成 `mocked=True`（已修复）

写作时 `llm.py:149` 命中缓存返回 `LLMResponse(..., mocked=True)`。但这个响应是**真实模型输出**（只是复盘），不是 mock；而 `LLMResponse.mocked` 的注释说「True when this came from a mock — the lens badges these differently」（`llm.py:36`），于是缓存命中会被 lens 徽章误标成「假模型」。

**修复**：现在返回 `mocked=False, usage={"cached": 1}`（`llm.py:156-157`），上方注释写明理由 —— 只有「这次没调模型」不同，标 `mocked` 会让徽章说谎。

**仍然要注意的语义**：`mocked=False` + `usage={"cached": 1}` 这个组合现在只能靠 `usage` 区分「真调了」和「读缓存」。`usage` 是自由 dict（`llm.py:32-38`），没有类型约束，消费方得自己约定 `cached` 键。`MockLLM` 那条路返回的 `mocked=True` 才是真正的 mock。

### 11.13 CLI 的 docstring 与实现不符（大部分已修复）

写作时的三个落差：docstring 只列 6 个子命令（漏 `checkpoints`）、`state` 的参数顺序写成 `<ckpt_id> <file>`（实现是 trace 在前）、`--at` 完全没提。

**修复**：`cli.py:1-19` 的 docstring 现在是 `Commands::` 块，**列全 7 个子命令**（show / dag / summary / checkpoints / state / run / watch），`state` 的两种用法都写了：`state traces/<file>.jsonl [<ckpt_id>]` 和 `state traces/<file>.jsonl --at <superstep>`，并明确「`state` takes the trace first」。注册点：show `:187`、dag `:195`、summary `:200`、checkpoints `:205`、state `:210`、watch `:218`、run `:223`。

**另有一处已修复**：docstring 里原本 `run` 写在 `watch` 之前，而 `build_parser` 先注册 `watch`（`:218`）再注册 `run`（`:223`），`--help` 按注册顺序排，所以列表与 docstring 最后两条顺序相反。已把 docstring 调成 `watch` 在前，现在两处一致。

**还有一处曾影响可读性的缺陷**：`--help` 的说明文字原本被 argparse 默认 formatter 重排成一整段无换行的文本（所有换行被吃掉），命令列表挤在一起无法阅读。已把 `build_parser()` 改用 `formatter_class=argparse.RawDescriptionHelpFormatter`，保留 `__doc__` 里的换行。

另外 `_load(path)` 现在显式处理 `path is None`（`cli.py:42-50`），因为 `state` 的 trace 参数是 `nargs="?"`；没有这个分支 `os.path.exists(None)` 会抛 `TypeError`。这条是对的，只是值得知道 `state` 可以不带 trace 调用、然后收到一句人话报错。

### 11.14 `max_concurrency` 记的是配置值不是实际值（已修复）

写作时 `RunResult.max_concurrency` 直接等于 `max(1, cfg.max_concurrency)`，而实际 worker 数是 `min(max_concurrency, len(plan))`（`runtime.py:331`），所以 `max_concurrency=4` 的 2 节点 run 也报 4。字段名让人以为是「这一步真的并发了几个」。

**修复**：`RunResult` 现在有两个字段（`runtime.py:70-73`）：

- `max_concurrency` —— **实际观测到的**单步最大并发。观测点是 `observed_conc = max(observed_conc, min(max_conc, len(plan)))`（`runtime.py:231`），初值 0（`:222`），结果写回 `max_concurrency=observed_conc`（`:308`）与 `allowed_concurrency=max_conc`（`:309`）。
- `allowed_concurrency` —— 配置上限。

重新实测：`max_concurrency=4` 的 2 节点 run → `max_concurrency=1`、`allowed_concurrency=4`。

**仍然要注意**：两者都是「计划的节点数」，不是「真实同时在跑的线程数」——`observed_conc` 由 `len(plan)` 算出，不含任何运行时测量。计划 4 个节点但其中 3 个秒退、1 个跑很久，仍会报 4。它是「意图并发度」，比旧版诚实一层，但不是火焰图。

### 11.15 `State.diff` 的键顺序随 hash seed 变化（已修复）

**问题**：`State.diff`（`state.py`）原本把两边通道名并成一个集合并遍历——`for name in set(self.values) | set(other.values)`。`set` 的顺序在同一个进程里稳定，但**跨进程取决于 `PYTHONHASHSEED`**。而 `diff` 会被序列化进 `state_merge` 事件、进而写进 trace 文件，所以同一份输入跑六次，trace 里 `messages` / `steps` 的先后可能是 `tam`、`atm`、`amt`、`tma` 六种不同排列之一。

**为什么这是真问题**：它直接违反第 1 节的原则「trace 是唯一事实来源」。两份内容完全相同的 trace 会因为键序不同而 diff 出噪音，于是「回放这条路径看到的东西」和「当初看到的东西」不再逐字节对得上。

**修复**：`State.diff` 改为**已声明 channel 按 schema 声明顺序排在前，其余（未声明的保留通道等）按字母序排在后**。这样 diff 的阅读顺序与状态定义顺序一致，且与 hash seed 无关。

**验证的诚实局限**：旧实现的 `set` 顺序在某些 seed 下恰好等于 schema 顺序（实测 `PYTHONHASHSEED=0/999` 不触发、`1/7` 触发），所以单进程内的回归测试只在这些 seed 上会红。回归测试 `test_diff_key_order_is_schema_order_so_traces_are_reproducible` 用 `zeta/alpha/mid` 三个 channel 断言「恰好是 schema 顺序、明确不是字母序」，这是单进程里能给出的最强断言。

### 11.16 `follow_trace` 会吃掉半行（已修复）

**问题**：`follow_trace` 是一个「读文件 → 到 EOF → sleep → 再读」的生成器。旧实现每次读到一个非空行就 `pos = fh.tell()` 然后 `Event.from_json(line)`。当写入方刚 flush 了 JSON 但还没写 `\n`（`json.dumps` 与 `write("\n")` 是两次 syscall），读者会读到一段**没有换行的残片**：`from_json` 抛 `JSONDecodeError`，而此时 `pos` 已经推进到残片之后——于是这半行**永久丢失**，等写入方补上 `\n` 和后半段时，读者从残片之后开始读，读到的是一个孤零零的 `}`。

**为什么这是真问题**：它不是「偶尔崩一下」，而是**静默丢事件**。丢掉的正好是 `run_end` 这种单条事件，现象是 `lens watch` 永远停在「运行中」。

**修复**（`events.py`）：只消费完整行——`if not line.endswith("\n"): break`，此时 `pos` 停在残片之前，下一轮重新读它整行。`serve.py:TraceFeed` 用的是同一套规则（`if not line.endswith("\n"): break`，在它自己的 `__iter__` 里），因为它面对的是同一类写入方；详见 §12.9。

**同一个助手还有第二个冻结模式（同一处，也修了）**：`pos` 是**字节偏移**，只对产生它的那个文件成立。如果文件被截断或轮转，新文件比旧偏移短，`fh.seek(pos)` 落在新 EOF 之后——跟随者从此**永远**读不到新事件，同样没有任何异常提示。判据是「变小的文件是*另一个*文件」：每轮 `os.path.getsize(path)`，`size < pos` 就把 `pos` 归零。M2 的 `serve.py`（`TraceFeed`，DESIGN 12.9 第 4 条）先修的这个，`events.py` 这一侧当时被标成「范围外」；收尾时补上，因为 `lens watch` 走的就是这条路径。

**验证**：五条回归测试——`test_follow_trace_skips_a_half_written_line_then_reads_it_whole`、`test_follow_trace_ignores_blank_lines_and_keeps_reading`、`test_follow_trace_recovers_when_the_file_shrinks_under_it`、`test_follow_trace_restarts_when_a_grown_rewrite_lands_mid_line`、`test_follow_trace_does_not_duplicate_after_one_failed_stat`，外加 `test_last_seq_in_file_finds_the_tail_record`。变异过程本身是一次性脚手架，没有进仓库（`.probe_*` 与 `.lens-test-tmp/` 都在 `.gitignore` 里，写进文档等于给读者一个他复现不了的「证据」）。要复现只需手工改一处再跑套件：删掉 `if not line.endswith("\n"): break`，半行那条必然变红；删掉 `offset_is_line_boundary` 的调用，长大改写那条必然变红；把 `except OSError:` 分支里的 `rotation.forget()` 改成 `pos = 0`，瞬时 stat 失败那条必然变红。**注意**：这类脚本会改源码，必须在 `finally` 里恢复，而且要先恢复再判读结果——否则一次超时就把工作树留在半改状态（这一条是踩过的）。

**一个测试写法上的教训**：截断这条测试必须把跟随者放到**工作线程**上驱动。没有修复时它不抛异常、只是再也不 yield，于是写在循环里的超时判断永远轮不到——测试不是变红，而是把整个套件挂死。挂死不是红灯。改成 `worker.join(3.0)` + `assert_true(not worker.is_alive(), ...)` 之后，变异测试才给出 `21/22` 的确定结果。

---

## 12. 浏览器 lens（M2）

### 12.1 一句话

`lens serve <trace>` 起一个本地 HTTP 服务，把**同一份 trace** 用 SSE 推给一个单文件页面。页面画三样东西：图（谁在跑）、superstep 时间轴（跑到哪了）、检查器（状态/差分/写入/事件/checkpoint）。

### 12.2 为什么是 SSE 而不是 WebSocket

问题的形状是**单向**的：trace 已经发生，浏览器只读。SSE 就是「一个 Content-Type 特别的 HTTP 响应体」，不需要握手、不需要帧库、不需要新依赖——`EventSource` 是浏览器基线 API。WebSocket 会多一个协议和一条代码路径，换不到任何能力。

实测过可行性再决定的：`ThreadingHTTPServer` + `text/event-stream` 下，帧以 0.141 / 0.192 / 0.243 秒的间隔逐步到达（服务端每帧故意 sleep 50 ms），**是真实流式，不是缓冲后一次性吐出**。

### 12.3 线格式

一帧一个事件，`id` 直接用 trace 自己的 `seq`：

```
id: 7
event: lens
data: {"seq":7,"kind":"node_end",...}
```

因为 `seq` 稠密且单调（`Emitter` 在锁内分配，见原则一的测试），浏览器的自动重连会带 `Last-Event-ID: 7`，服务端从 8 续传。于是**重连既不重复也不漏事件，而且服务端不需要任何会话状态**。

`data:` 里就是 `Event.to_json()` 的字节，不是第二种序列化——所以浏览器解析的和 JSONL 文件里的是同一串字节，两者不可能漂移。

### 12.4 两种模式，一条代码路径

- **replay**（默认）：读到 EOF，发 `done`，关连接。一份跑完的 trace 于是变成一个可拖动的录像。
- **follow**：到 EOF 不退出，每 100 ms 重读文件，新行来了就推。运行中的 run 是活的。

两者是同一个生成器，`follow` 只决定「到 EOF 停不停」。

### 12.5 页面的结构：一个纯函数 + 一个画匠

`web/index.html` 里只有两个角色：

- `derive(events, upto)` —— 输入「事件数组 + 看到第几条」，输出一个完整的视图模型（状态、diff、每条边的使用次数、每个节点的访问次数、当前 superstep…）。**它是纯的**：不碰 DOM、不读全局可变状态、不看时钟。
- `paint()` —— 把视图模型写进 DOM。唯一有副作用的地方，并且被 `requestAnimationFrame` 合并。

拖动条往回拖时，页面**不是「撤销」**，而是用更短的前缀重新跑一遍 `derive`。这是刻意的：撤销需要维护逆操作，而重放一个前缀只需要一个纯函数。这也是原则二在 UI 层最直接的一次兑现——**时间旅行在这个页面里是免费的，因为它本来就没有「当前状态」之外的东西**。

五个检查器面板与它们的数据来源：

| 面板 | 内容 | 数据来源 |
| --- | --- | --- |
| `state` | 全部 channel，变更的标 ●，附完整 JSON | 最近一条 `state_merge.data.state` |
| `diff` | 每个 channel 的 before/after 两栏 + 原始 patches | `state_merge.data.diff`（**trace 自带，页面不重算**） |
| `writes` | 每个节点交给 reducer 之前的 writes + 耗时 + 流式 token | `node_end.data.writes` / `node_token` |
| `events` | 全量事件列表，按 kind 着色，过去的事件变暗 | 事件数组本身 |
| `checkpoints` | 时间轴上的点，按 status 着色 | `checkpoint_created`（按 id 去重，留最后一条） |

### 12.6 图是怎么画的：只吃 `run_start.data.graph`

页面**不 import 图定义，也不知道任何节点名**。它拿 `run_start.data.graph` 的快照（`name` / `entry` / `nodes` / `edges` / `conditional_edges` / `channels`）现算布局：

- 从 `entry` 开始 BFS 分层，环上的边变成「回边」（所以 tool-loop 的 `agent ⇄ tools` 画得出来）
- 不可达节点放到最后一层之后
- 任何指向 `__end__` 的边或候选，都会补一个终点框

布局之后是**边路由**，这是整份文件里唯一真正难的地方。朴素的中点连线会在三种情况下出错，都是实测出来的：

1. **互逆的一对边**（`tools→agent` 和 `agent→tools`）会算出同一条贝塞尔——两条边完全重叠，两个标签也落在同一个坐标上互相盖住
2. **跳层的边**（`agent→__end__`，中间隔着 `tools`）会从 `tools` 的框里穿过去
3. **自环**（节点路由回自己）根本没有合理的中点

所以路由规则是按这三类分别写的：相邻层向前走水平线、跳层向下绕一个 `nadir`、向后或同层的从顶部绕过去、自环从自己的框上方翻过去。**没有两条边共用一条线，也没有边穿过任何一个框。**

还有一个只有新旧 trace 混读才会暴露的坑：**`conditional_edges[].candidates` 是路由函数的返回值，不是节点名。** 新快照同时带 `mapping`（返回值 → 节点）和人读的 `descriptions`；早期写下的快照没有 `mapping`，`descriptions` 里存的就是节点名。页面用一个 `candTarget()` 统一解析——先 `mapping`、再退回 `descriptions`、最后原样返回——所有调用点（BFS 分层、`hasEnd` 判定、边构造、`route_decision` 高亮）都走它。

这个坑值得单独写出来，因为**它在那三份历史 trace 上是看不出来的**（它们恰好是旧格式，`descriptions` 就是节点名，所以「直接用 descriptions」看起来完全正确）。一个只按历史 trace 目视验证过的实现，会在新跑出来的 trace 上把边指向一个不存在的节点，于是边消失、目标节点丢层。`tests/test_serve.py` 里 `test_page_resolves_candidates_through_mapping_not_prose` 就是钉这个的，包括断言解析器读过 `ce.mapping`、以及禁止旧的错误写法。

### 12.7 页面遵守原则二到什么程度

这不是口头承诺，`tests/test_serve.py` 里有一条测试直接扫页面源码，断言它不含 `lens_harness` / `runtime.py` / `graph.py` / `checkpoint.py`，也不含 `state_merge(` / `superstep_start(` 这类「偷偷调用内核」的痕迹；另一条断言它没有任何外链（`<script src>`、`<link href>`、`@import`、第三方 URL），所以它真的能在断网、无 CDN 的情况下打开。

还有一条断言页面**覆盖了 `EVENT_SCHEMA` 里的每一种事件**——将来加了第 12 种事件却忘了处理，测试会红，而不是页面静默地少显示一个东西。

### 12.8 这一层明确不做的事

- **不做写操作**：不能跑图、不能 resume、不能提交审批。`serve.py` 只注册了 `do_GET`。
- **不做多 trace**：一个进程一个文件。要对比两次运行，开两个端口。
- **不做前端工程化**：没有 React、没有打包器、没有 TypeScript。单文件、内联、零依赖——这样「一个 trace 长什么样」这件事的全部代码能在一个下午读完。
- **不做增量状态协议**：服务端推的是完整事件，页面重放前缀，不做补丁叠加（见第 10 节那条澄清）。

### 12.9 传输层的四个「看起来能跑」的坑

第一版 `serve.py` 在浏览器里工作得完全正常：49 个事件、图、时间轴、diff 全对。但它在四个地方是错的，而且四个都**测不出来**——因为 `urllib` 自己会替客户端加上 `Connection: close`，因为线程要 20 秒后才泄漏得看得见，因为它们只在异常路径上发生。这一节记下来，是因为「在浏览器里看着对」正是这个项目最不该当作证据的东西。

1. **流不结束**。`text/event-stream` 没有 `Content-Length`，所以 EOF 是客户端唯一能知道流结束的信号。第一版发完 `done` 就 `return`，HTTP/1.1 keep-alive 把 socket 留着，浏览器就一直等一段不会来的字节。修法：`self.close_connection = True` 加显式 `Connection: close`。复现方式值得记住——`curl -N --http1.1` 会挂到超时，而任何基于 `urllib` 的测试永远是绿的。
2. **生产线程泄漏**。生产者线程做两件会无限等待的事：在 `poll_s` 上空轮询一个安静的文件，以及往一个满了的 `Queue` 上阻塞 `put`。用户关掉标签页之后，两处都不知道该走了，于是每个被遗弃的连接留下一个线程和一个文件句柄。修法：把 `stop` 事件穿进轮询循环（`TraceFeed` 新增可选关键字参数 `stop`，默认 `None`，保持向后兼容），并把 `out.put(event, timeout=0.5)` 放进一个反复检查 `stop` 的循环里。
3. **截断伪装成完成**。解析不了的行会抛异常，第一版发一个 `error` 帧然后**照样**发 `done`。页面把「replay complete」挂在 `done` 上，于是 2/49 条事件的截断流看起来是一次干净的结束——对一个「老实记录发生了什么」的工具来说这是最坏的失败方式。修法：出错就不发 `done`，并在 `error` 的载荷里写 `truncated: true`。
4. **轮转后静默冻结**。`TraceFeed` 记字节偏移；文件变小（重跑、轮转）之后 `seek(pos)` 落在 EOF 之外，读不到任何东西，也不报错。修法：每轮 `stat` 一次，`size < pos` 就把 `pos` 归零。注释里写的是「变小的文件是*另一个*文件」。

另外三个小的：`_client_after()` 原来不校验游标（`?after=abc` 静默从头重放，`?after=1e9` 静默返回 0 条事件），现在 400；`make_server` 原来只 `os.path.exists`（目录也收，然后流到一半才报错），现在要求可读的**文件**；`/api/health` 原来漏绝对路径，现在只回 basename；**meta 帧不同**——它同时给出 `trace`（basename，给人看）和 `path`（完整路径，给已连接的一方定位用），这两处刻意不一致，不要「统一」成一样。

这一轮之后 `tests/test_serve.py` 从 22 条涨到 28 条，新增的六条分别钉住上面四条和游标校验、目录拒绝。收尾时第二轮评审又找出三处轮转相关的漏洞，于是再涨到 31 条（见 §12.10）。它们的共同点是**都能在旧代码上红**——其中 `test_events_socket_reaches_eof_after_done` 必须绕开 `urllib` 直接开原始 socket，否则它永远看不到那个 bug。

### 12.10 轮转：文件在脚下被换掉

§12.9 修的是「服务端把流管好了」。第二轮评审问了一个更难的问题：**如果 trace 文件在读者脚下被换掉呢？**
这在教学项目里不是边角情况——`lens watch` 和 `lens serve --follow` 最常见的用法就是「跑一个例子，看它」；
再跑一次，文件就被重写了。三次评审一共找出六个变体，前四个在 §12.9 之后就已经不成立，后两个是这一轮补的。

把「读取位置」这件事想清楚只需要一句话：**`pos` 是字节偏移，而字节偏移只对产生它的那个文件有意义。**
于是判断「还是不是同一个文件」不能只看长度，因为：

| 文件发生了什么 | `pos` 会怎样 | 只看长度的后果 |
| --- | --- | --- |
| 被追加（正常运行） | 仍在合法位置 | 正确，且这是最常见的路径 |
| 被截短 | 落在 EOF 之外 | 永远读不到新内容，不报错 |
| 原地重写成**一样长** | 落在行中间 | 不会触发任何长度判断，静默读到旧内容的尾巴 |
| 原地重写成**更长** | 仍在新文件内，但在行中间 | `readline` 返回半条记录，`JSONDecodeError` |
| 被替换（`rename`） | 属于另一个 inode | 无 |

所以规则被抽成一个共享对象 `events.TraceRotation`，两个读者（`follow_trace` 与 `serve.TraceFeed`）用的是**同一份判断**，
而不是各自抄一遍——第二轮评审的着力点正是「两份实现会漂移」。它记三样东西：`st_ino`、`st_size`、`st_mtime_ns`。
三个条件缺一不可：inode 变了 = 被替换；变短了 = 被截断；长度没变但 mtime 动了 = 原地等长重写（确定性的例子重跑正好产生这个）。
正常的追加会让文件变长，因此不可能误触发第三条。

还有两个坑不在「长度」这个维度上：

**一次 `stat` 失败不等于文件空了。** 第一版把 `except OSError` 当成 `size = 0`，于是 `size < pos` 对任何已读过字节的读者都成立——
一次瞬时错误就把整份 trace 当新文件重读一遍，事件变成 `[0,1,2,0,1,2]`。正确做法是 `rotation.forget()` 然后照常轮询：
没测到的文件不能拿来当证据。

**变长重写仍会落进行中间。** `st_size` 比旧偏移大，「变短」不成立，`seek(pos)` 完全合法，但那个位置已经是别的行了。
判据只能看内容和位置的关系：`events.offset_is_line_boundary(path, pos)` 读 `pos-1` 那一个字节，不是换行就从头来。
追加永远满足这一条，重写永远不满足，正好把两者分开。

**恢复游标也要跟着重置。** 网页重连时带 `Last-Event-ID`，它的含义是「这份文件里 N 之前我都看过了」。
如果文件已经被重跑替换，新文件从 0 开始、永远到不了 N，那么老老实实尊重游标的客户端就什么都收不到——
而且看起来不像故障，像「这一轮还没开始写」。`events.last_seq_in_file(path)` 从文件尾部反向读最后一条完整记录，
把这个 N 跟文件真实的尾号对一次，对不上就丢弃游标。反向读而不是全文扫描，是因为它每个连接只跑一次，而 trace 可以很大。

**页面上同一件事的另一半：** 服务器从头重发之后，浏览器收到的 seq 会**比已有的更小**。
`push()` 因此把「seq 没有前进」当作「这是一次重跑」，直接清空事件列表重新开始——这是安全的，因为页面本来就是事件前缀的纯函数，
「忘掉再重放」与「重新打开」等价。不这么做的话，计数器会显示 `98 / 98 events`（两轮拼在一起），而最新 seq 是 48，
滑动条的刻度（按数组下标）和它的标签（打印 seq）就会互相矛盾。

**这一轮新增的 7 条测试**（内核 +3，传输 +4）：瞬时 stat 失败不重复、等长重写被识别、变长重写从行首重来、
`last_seq_in_file` 尾记录识别（含半行尾）、游标不掩盖重跑、页面在 seq 回退时重开模型、游标只接受 ASCII 数字。
其中「变长重写」和「游标掩盖重跑」两条都做过变异验证：把对应机制删掉，各自的那条测试确定变红（`24/25`、`28/29`）。

顺带修掉一条真实存在的不精确：`_parse_cursor` 原来用 `body.isdigit()`，而 `"٤٠".isdigit()` 是 `True`、`int()` 也能算出 40，
非 ASCII 数字因此被接受，与 docstring 承诺的「只要 ASCII 数字」相反。现在是 `body.isascii() and body.isdigit()`。
---

## 13. trace 对比（M3）

### 13.1 一句话

`lens diff a.jsonl b.jsonl` 回答：**归一化掉每次运行必然不同的东西之后，这两次运行的行为差在哪？** 浏览器版是 `lens serve a.jsonl b.jsonl` 的 `/compare` 页。

### 13.2 为什么裸 diff 没有用

README 早就实测过：两份 trace 的 `ts`、`run_id`、`checkpoint_id`、`parent_id`、`duration_ms` 这 5 个字段**必然**不同，所以 `diff` 永远非空。M1/M2 的视图回答「一次运行里发生了什么」，M3 回答的是另一个方向的问题——「两次运行差在哪」，这要求先定义「什么不算差」。

### 13.3 归一化（`compare.py`）

归一化是契约的核心，规则只有三条：

1. **丢弃**：`ts`（时序）、`run_id`（身份）、`duration_ms`（机器噪音）。丢掉它们是「diff 为空 ⟺ 行为相同」成立的前提。
2. **占位**：`checkpoint_id` / `parent_id` → `ss<N>`（N = checkpoint 的 superstep）。真实 id 含 6 位随机 hex，但「第 2 步产出的 checkpoint」是故事结构里的稳定位置。
3. **聚合**：`node_token` 逐块事件 → 每 `(superstep, node)` 一条整段文本。逐 chunk 对齐 diff 出的是切块粒度（`chunk_size` 一变就全红），不是模型说了什么。

### 13.4 对齐

每条记录拿一个结构键，按首见顺序配对：

| 事件 | 键 |
| --- | --- |
| `run_start` | `(kind, 序号)` —— 含 resume 的 trace 有两条，按出现顺序配 |
| `superstep_start` | `(kind, index)` |
| `node_start` / `node_end` / `node_error` | `(kind, superstep, node)` |
| `route_decision` | `(kind, superstep, source)` |
| `state_merge` | `(kind, superstep)` |
| `checkpoint_created` | `(kind, 占位 id, status)` |
| tokens 聚合 | `("tokens", superstep, node)` |
| 其余（`interrupt`、`error_reported`…） | `(kind, superstep, node)` |

对齐结果是四类：`same`、`changed`（列出具体字段）、`only-a`、`only-b`。

**已知的结构性余波（不是 bug）**：两个 run 结束位置不同时，`run_end`（键含 supersteps 数）和终态 checkpoint（superstep 号不同）会各自出现在 only 侧。节点级事件不受影响——`test_only_on_one_side_is_reported` 专门钉住「a 是 b 的严格子集时，only-a 里不得出现任何节点级事件」，防止将来有人把余波当 bug 修掉。

### 13.5 两个视图，一个内核

与 M1/M2 同构：`compare.py` 是纯函数内核（只 import `events`，原则二），两个消费者并列：

- **终端**：`cmd_diff` → `render_diff`，有差异退出码 1（GNU diff 约定，方便脚本）。
- **浏览器**：`/api/diff` 把对比结果序列化成一个 JSON 文档，`web/compare.html` 取它、渲染它。页面不 import 内核、不做任何计算——对比在服务端做完了，页面只画判决。每次请求现算不用缓存：trace 文件可能被重跑改写，缓存的 diff 会对现在文件里的内容说谎。

单 trace 启动时 `/compare` 是 404、`/api/diff` 是 504，都带人话消息——静默失败在这里同样不可接受。

### 13.6 验收测试（`tests/test_compare.py`，8 条）

- `test_identical_runs_diff_clean` —— **同一个图跑两次，diff 必须为空**。这是整条推理链的地基；若它红了，归一化就漏了噪音字段。
- `test_checkpoint_ids_become_stable_placeholders` —— uuid 后缀不得漏进键或数据。
- `test_token_events_aggregate_into_one_record` —— 切块方式不进 diff。
- `test_changed_state_is_found_and_named` —— changed 记录必须列出字段，且字段里不得有噪音。
- `test_only_on_one_side_is_reported` —— 见 13.4 的余波说明。
- `test_node_error_differences_are_visible` —— 失败 vs 成功的 run，`node_error` 必须出现在 diff 里。
- `test_render_diff_is_readable_and_respects_max_only` —— identical 一行、差异分节、`--max-only` 截断。
- `test_resume_trace_diffs_clean_against_itself` —— 含 invoke+resume 的 trace 对自己 diff 必须干净（钉住多 `run_start` 配对逻辑）。

---

## 附：如何跑起来

```bash
# 内核自测（30 个）+ 流程自测（8 个）+ lens serve 自测（31 个）+ trace 对比自测（8 个）
py -3.13 -m tests.test_kernel
py -3.13 -m tests.test_flows
py -3.13 -m tests.test_serve
py -3.13 -m tests.test_compare
py -3.13 -m tests.test_dsl

# 渲染一份现有 trace
py -3.13 -m lens_harness show traces/r_70b323a6.jsonl

# checkpoint 结构 / 事件计数 / checkpoint 清单
py -3.13 -m lens_harness dag traces/r_70b323a6.jsonl
py -3.13 -m lens_harness summary traces/r_70b323a6.jsonl
py -3.13 -m lens_harness checkpoints traces/r_70b323a6.jsonl

# 某个 checkpoint 的状态（注意：trace 在前、id 在后；值是摘要，见 11.4）
py -3.13 -m lens_harness state traces/r_70b323a6.jsonl demo_0003_acd875
# 或者按 superstep 取（--at，见 11.13）
py -3.13 -m lens_harness state traces/r_70b323a6.jsonl --at 3

# 跑范例并写出一份新 trace
py -3.13 -m lens_harness run examples.agent:build --input '{"question":"What is 17*23?"}'

# 跟随一段正在写入的 trace
py -3.13 -m lens_harness watch traces/<run_id>.jsonl

# 在浏览器里看同一份 trace（第 12 节）
py -3.13 -m lens_harness serve traces/r_70b323a6.jsonl
# 边跑边看
py -3.13 -m lens_harness serve traces/<run_id>.jsonl --follow
```

依赖：**零**。纯标准库，Python 3.13。本机 `python`/`python3` 是 Windows Store 占位程序（退出码 9009），必须用 `py -3.13`。
