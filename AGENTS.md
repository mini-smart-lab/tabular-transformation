# AGENTS.md — Hướng Dẫn Dành Cho Coding Agent

Tài liệu này cung cấp hướng dẫn toàn diện cho các Coding Agent (Antigravity, Claude Code, Cursor, Copilot, Aider, v.v.) khi đọc hiểu, bảo trì, mở rộng và kiểm thử codebase **Tabular Transformation Discovery Harness**.

---

## 1. Tổng quan dự án (Project Overview)

**Tabular Transformation Discovery Harness** là một research harness 2 tầng (Two-tier evaluation harness) dùng để đánh giá khả năng khám phá và suy luận quy luật biến đổi dữ liệu dạng bảng (tabular column transformations) của các mô hình ngôn ngữ lớn (LLM):

- **Tier 1 — Discovery (Tầng Khám phá):** LangGraph agent nhận đầu vào duy nhất là `train_before.csv` và `train_after.csv` trong một sandbox độc lập. Agent có 2 tools:
  - `run_python(code)`: Chạy thử code phân tích/thực nghiệm trong sandbox.
  - `submit_solution(script, transformations)`: Nộp hàm `transform(df)` hoàn chỉnh cùng danh sách toán tử đã phát hiện.
  - *Kiểm soát ngân sách nghiêm ngặt (Hard Budget):* Token I/O, số lượt tool call, execution wall-time, CPU time và episode wall-time.
- **Tier 2 — Evaluation (Tầng Đánh giá Độc lập):** Chạy hàm `transform(df)` đã nộp trong một sandbox hoàn toàn mới trên tập dữ liệu ẩn `hidden/test_before.csv`. Kết quả dự đoán được so sánh với `hidden/test_after.csv` qua hệ thống chấm điểm 2 chiều (Black-box functional correctness & White-box symbolic recovery).

---

## 2. Cấu trúc thư mục & Các module trọng tâm

```text
tabular-transformation/
├── configs/                  # Cấu hình YAML (harness.yaml, nvidia.yaml, openai.yaml)
├── docs/                     # Tài liệu kiến trúc chi tiết (docs/harness_architecture.md)
├── notebooks/                # Dữ liệu mẫu (processed_data.csv) và notebook nghiên cứu
├── pipeline_runs/            # Thư mục lưu kết quả thực thi pipeline (tự động sinh)
├── src/                      # Source code chính của harness
│   ├── agent/                # LangGraph State Machine, State, Prompts, Schemas, Budget & Token usage
│   │   ├── budget.py         # Adaptive Tiered Budget Controller
│   │   ├── graph.py          # LangGraph discovery workflow (agent_node, tool_node, routing)
│   │   ├── prompts.py        # System prompt & guidance cho LLM
│   │   ├── schemas.py        # Pydantic schemas cho tool submission & validation
│   │   ├── state.py          # AgentState definition
│   │   └── token_usage.py    # Theo dõi và tính toán token tiêu thụ
│   ├── benchmark/            # Quản lý task, runner, suite và tổng hợp kết quả
│   │   ├── aggregate.py      # Gộp các file result JSON thành summary.csv
│   │   ├── result.py         # Cấu trúc dữ liệu kết quả benchmark
│   │   ├── runner.py         # Single-task runner (chạy 1 task cụ thể)
│   │   ├── suite.py          # Multi-task suite runner (chạy hàng loạt task)
│   │   ├── task.py           # Task data loader & model
│   │   └── task_builder.py   # Bộ sinh task theo cấp độ khó Level 0-5
│   ├── evaluator/            # Đánh giá độ chính xác (Functional & Operator metrics)
│   │   ├── column_metrics.py # Transformed column accuracy, per-column match
│   │   ├── evaluator.py      # Evaluator engine tổng hợp
│   │   ├── operator_metrics.py# Operator precision/recall/F1, sequence accuracy
│   │   └── validation.py     # AST verification cho submit_solution script
│   ├── logging_utils/        # Xử lý logging & trajectory JSONL
│   ├── sandbox/              # Cơ chế cách ly thực thi mã
│   │   ├── base.py           # Sandbox interface chuẩn
│   │   ├── docker.py         # Docker container sandbox
│   │   ├── local_process.py  # Subprocess sandbox với sanitized env & timeout
│   │   └── models.py         # ExecutionResult, SandboxConfig
│   ├── tools/                # Định nghĩa công cụ LangChain cho Agent
│   │   └── agent_tools.py    # run_python và submit_solution
│   ├── config.py             # Parser cấu hình harness YAML & CLI
│   └── model_factory.py      # Registry và khởi tạo ChatModel (NVIDIA, OpenAI, Extensible)
├── tests/                    # Toàn bộ Pytest suite (kiểm thử độc lập không cần API key/Docker)
├── transformation/           # Thư viện 24 phép biến đổi chuẩn hóa
│   ├── base.py               # Abstract base class `Transformation`
│   ├── conditional.py        # Piecewise / Indicator transformations
│   ├── exp.py                # Exponential transformation
│   ├── interaction.py        # Multi-column interaction transformations (Add, Sub, Mul, Div, Cond)
│   ├── log.py & log1p.py     # Logarithmic transformations
│   ├── power.py              # Power, Cube, CubeRoot transformations
│   ├── scaling.py            # StandardScaler, MinMaxScaler
│   ├── sqrt.py & square.py   # Sqrt, Square transformations
│   └── trig.py               # Sin, Cos, Tanh, Sigmoid transformations
├── .env.example              # Mẫu biến môi trường API keys
├── requirements.txt          # Python dependencies
└── run_pipeline.py           # Entrypoint CLI chạy toàn bộ pipeline end-to-end
```

---

## 3. Môi trường phát triển & Lệnh thường dùng

### 3.1. Thiết lập môi trường (Windows PowerShell)
- Python runtime được cấu hình trong virtual environment `.venv`.
- Luôn sử dụng `.venv\Scripts\python.exe` để chạy các script và lệnh test.

```powershell
# Cài đặt thư viện phụ thuộc
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Cấu hình biến môi trường
Copy-Item .env.example .env
# Điền API keys cần thiết (NVIDIA_API_KEY, OPENAI_API_KEY) vào .env
```

### 3.2. Chạy kiểm thử (Testing)
Tất cả 42+ tests được thiết kế để chạy **offline, nhanh chóng, không cần API Key và không phụ thuộc Docker**.

```powershell
# Chạy toàn bộ test suite
.venv\Scripts\python.exe -m pytest -q

# Chạy một file test cụ thể
.venv\Scripts\python.exe -m pytest tests\test_task_builder.py -v

# Chạy test với output chi tiết khi debug
.venv\Scripts\python.exe -m pytest -vv -s
```

### 3.3. Chạy Pipeline Benchmark
```powershell
# 1. Chạy sinh task thử nghiệm (không gọi LLM, không tốn credit)
.venv\Scripts\python.exe run_pipeline.py --source notebooks\processed_data.csv --prepare-only --level 3 --task-count 5 --seed 42

# 2. Chạy full pipeline end-to-end với LLM (chạy song song 4 workers)
.venv\Scripts\python.exe run_pipeline.py --level 3 --task-count 24 --workers 4 --seed 42

# 3. Chạy benchmark trên 1 task đơn lẻ
.venv\Scripts\python.exe -m src.benchmark.runner --task data\0 --model-config configs\nvidia.yaml --harness-config configs\harness.yaml
```

---

## 4. Nguyên tắc cốt lõi & Ranh giới an toàn (Strict Guardrails)

Khi chỉnh sửa hoặc mở rộng codebase, Agent **BẮT BUỘC** tuân thủ các nguyên tắc sau:

### 4.1. Cách ly dữ liệu & Sandbox (Data & Security Isolation)
1. **Tuyệt đối không để lộ dữ liệu ẩn (Hidden Test Set) cho Tier 1:**
   - Discovery Sandbox chỉ được phép mount `train_before.csv` và `train_after.csv`.
   - `hidden/test_before.csv` chỉ được mount ở Evaluation Sandbox (Tier 2).
   - `hidden/test_after.csv` chỉ được đọc bởi `Evaluator` ở host, **không bao giờ** được mount vào bất kỳ sandbox nào.
2. **Không truyền biến môi trường nhạy cảm vào Sandbox:**
   - `LocalProcessSandbox` và `DockerSandbox` phải luôn scrub môi trường (loại bỏ toàn bộ `API_KEY`, `TOKEN`, host secrets).
   - Host API keys chỉ được sử dụng trực tiếp bởi client LLM ở host process.

### 4.2. Tính tái lập thí nghiệm (Strict Reproducibility)
- Mọi logic lấy mẫu ngẫu nhiên (data sampling, task generation, operator selection) phải phụ thuộc vào tham số `seed` (mặc định: `42`).
- Không sử dụng `random` hay `np.random` không kiểm soát seed.
- Khởi tạo LLM model phải truyền đúng `seed` và `temperature: 0.0`.

### 4.3. Quản trị ngân sách thích ứng (Adaptive Tiered Budget)
- Hệ thống áp dụng ngân sách theo từng Level (Level 0 $\to$ Level 5) trong `src/agent/budget.py`.
- Nếu token, số tool calls, thời gian thực thi vượt quá giới hạn, `BudgetController` phải cắt ngắn episode một cách graceful và trả về trạng thái tương ứng (`exceeded_budget`, `timeout`), không để crash unhandled exception.

---

## 5. Hướng dẫn mở rộng chức năng (Extension Guidelines)

### 5.1. Thêm một Phép biến đổi mới (New Transformation Operator)
1. Kế thừa lớp [`Transformation`](transformation/base.py).
2. Triển khai phương thức:
   - `apply(self, df: pd.DataFrame) -> pd.DataFrame`: Logic biến đổi cột.
   - `is_applicable(self, df: pd.DataFrame) -> bool`: Kiểm tra điều kiện toán học (tránh chia 0, log số âm, NaN, Infinite, tràn số).
3. Đăng ký class mới trong `transformation/__init__.py` và cập nhật vocabulary trong `configs/harness.yaml`.
4. Viết unit test bổ sung trong `tests/test_transformations_and_levels.py`.

### 5.2. Thêm Provider LLM mới
1. Mở `src/model_factory.py`.
2. Sử dụng `register_provider(name, factory_fn)` để đăng ký provider mới trả về instance kế thừa `BaseChatModel` (LangChain).
3. Đảm bảo hỗ trợ tool calling / function calling và bind đúng schemas của `run_python` và `submit_solution`.
4. Tạo file cấu hình mẫu trong `configs/<provider>.yaml`.

### 5.3. Mở rộng Hệ thống Đánh giá & Metrics
- Toàn bộ metric được xử lý tại `src/evaluator/`.
- Phân tách rõ ràng giữa **Functional Column Accuracy** (chấm trên DataFrame output) và **Operator Metrics** (chấm trên AST biểu thức toán tử được submit).
- Khi thêm metric mới, cập nhật schema kết quả trong `src/benchmark/result.py` và bảng tổng hợp `src/benchmark/aggregate.py`.

---

## 6. Checklist kiểm tra trước khi hoàn thành công việc (Agent Pre-flight Checklist)

Trước khi xác nhận hoàn thành bất kỳ task nào, Agent cần thực hiện:

- [ ] **Chạy toàn bộ test suite:** `.venv\Scripts\python.exe -m pytest -q` đảm bảo 100% tests pass.
- [ ] **Kiểm tra cú pháp & Type Hints:** Không gây ra `SyntaxError` hoặc xung đột schema Pydantic.
- [ ] **Bảo toàn tính đóng gói Sandbox:** Đảm bảo không để lộ host environment hoặc test data.
- [ ] **Không tạo file rác:** Không commit các file tạm, `.pyc`, checkpoint chưa được dọn dẹp vào repo.
