# Kiến trúc Tabular Transformation Discovery Harness

Tài liệu này mô tả chi tiết và ngắn gọn về toàn bộ cấu trúc hệ thống, quy trình thực thi, cơ chế biến đổi và hệ thống đánh giá của Harness.

---

## 1. Sơ đồ luồng tổng quan (Workflow Architecture)

```mermaid
flowchart TD
    subgraph S1["1. Sinh Task (Task Builder)"]
        RawData["Dataset gốc (.csv)"] --> Split["Chia Train (Công khai) / Test (Ẩn)"]
        Split --> ApplyOps["Áp dụng Operators theo Level (1-5)"]
        ApplyOps --> TaskFiles["train_before.csv / train_after.csv<br/>hidden/test_before.csv / hidden/test_after.csv"]
    end

    subgraph S2["2. Tầng 1: Khám phá & Suy luận (Discovery Tier)"]
        TaskFiles -->|Chỉ mount train files| Agent["LangGraph Agent (LLM)"]
        Agent <-->|run_python| SandboxD["Sandbox Khám phá (Local/Docker)"]
        Agent -->|Quản lý ngân sách| BudgetCtrl["Budget Controller<br/>(Tokens, Tool calls, Time)"]
        Agent -->|submit_solution| Sub["Mã transform(df) + Danh sách Operator"]
    end

    subgraph S3["3. Tầng 2: Đánh giá độc lập (Evaluation Tier)"]
        Sub --> SandboxE["Sandbox Đánh giá Mới (Cô lập)"]
        TaskFiles -->|Chỉ mount test_before.csv| SandboxE
        SandboxE --> RunTrans["Chạy transform(test_before) -> predicted.csv"]
        RunTrans & TaskFiles --> Evaluator["Evaluator Engine"]
        Evaluator --> Metrics["- Transformed Column Accuracy<br/>- Functional Column Accuracy<br/>- Operator Precision / Recall / F1<br/>- Exact Sequence Accuracy"]
    end

    subgraph S4["4. Tổng hợp & Báo cáo (Reporting)"]
        Metrics --> Summary["results/*.json -> summary.csv"]
        Summary --> Reports["report.md & report.json<br/>(Benchmark Score & Conditional Score)"]
    end
```

---

## 2. Các thành phần cốt lõi

### 2.1. Hệ thống Phép biến đổi (`transformation/`)
Gồm 24 toán tử kế thừa từ lớp cơ sở [`Transformation`](../transformation/base.py), tích hợp hàm kiểm tra tính hợp lệ toán học `is_applicable` (chống tràn số, vô cùng, chia cho 0, log số âm):
- **Đơn biến & Phi tuyến:** `Log`, `Log1p`, `Square`, `SquareRoot`, `Cube`, `CubeRoot`, `Exp`, `Inverse`, `Abs`.
- **Chuẩn hóa (Scaling):** `StandardScaler` (Z-Score), `MinMaxScaler`.
- **Lượng giác & Kích hoạt:** `Sin`, `Cos`, `Tanh`, `Sigmoid`.
- **Phân mảnh & Điều kiện (Piecewise / Conditional):** `IndicatorZero` ($\mathbb{I}(x > 0)$), `IndicatorMean` ($\mathbb{I}(x > \text{mean}(x))$), `ClipZero` ($\max(x, 0)$ / ReLU), `PiecewiseLog` ($\log(x) \text{ if } x > 1 \text{ else } x$).
- **Tương tác đa biến (Inter-column):** `AddInteraction` ($A + B$), `SubtractInteraction` ($A - B$), `MultiplyInteraction` ($A \times B$), `DivideInteraction` ($A / B$), `ConditionalInteraction` ($A \text{ if } A > \text{mean}(A) \text{ else } B$).

### 2.2. Bộ sinh Task theo cấp độ khó & Độ sâu cây cú pháp (`src/benchmark/task_builder.py`)
- **Level 0 (Atomic):** 1 phép biến đổi đơn biến $f(x)$ ($d=1$).
- **Level 1 (Atomic Unary):** 2 phép biến đổi đơn biến độc lập ($d=1$, không lồng ghép).
- **Level 2 (Composed Unary):** 3 phép biến đổi lồng ghép $f(g(x))$ ($d=2$, bảo đảm lồng ghép chuỗi hàm).
- **Level 3 (Interactions):** 3 cột độc lập + 1 tương tác đa biến ($x_1 \odot x_2$, $d=2$).
- **Level 4 (Conditional / If-Else):** 3 cột độc lập (ưu tiên hàm phân mảnh/indicator) + 1 tương tác điều kiện phân mảnh.
- **Level 5 (Hybrid Deep AST):** 4 cột độc lập (lồng ghép + conditional) + 2 tương tác đa biến với độ sâu $d \ge 3$ ($f(x_1 \odot x_2)$, v.v.).

### 2.3. Môi trường Sandbox (`src/sandbox/`)
- **Nguyên tắc an toàn:** Không truyền API Key / Environment variables của host vào sandbox; làm sạch biến môi trường.
- **`LocalProcessSandbox`:** Thực thi code qua subprocess độc lập, áp timeout và giới hạn độ dài stdout/stderr.
- **`DockerSandbox`:** Khởi tạo container riêng biệt cho mỗi lượt chạy, cách ly network (`none`), giới hạn CPU (1 core) và RAM (1GB).

### 2.4. Quản trị Ngân sách Thích ứng (Tiered Budget - `src/agent/budget.py`)
Hệ thống tự động điều chỉnh ngân sách theo độ phức tạp của từng Level (có thể ghi đè bằng `--fixed-budget`):
- **Level 0 - 1:** 40,000 tokens, 8 tool calls, timeout 15s, 30s CPU, 120s wall time.
- **Level 2:** 60,000 tokens, 12 tool calls, timeout 15s, 60s CPU, 180s wall time.
- **Level 3:** 80,000 tokens, 15 tool calls, timeout 15s, 60s CPU, 240s wall time.
- **Level 4:** 100,000 tokens, 20 tool calls, timeout 15s, 90s CPU, 300s wall time.
- **Level 5:** 120,000 tokens, 25 tool calls, timeout 15s, 120s CPU, 360s wall time.

### 2.5. Tính Tái lập Thí nghiệm (Reproducibility)
- Tham số `--seed` (mặc định: `42`) được cố định cho cả:
  1. Bộ sinh task và lấy mẫu dữ liệu ngẫu nhiên xác định.
  2. Khởi tạo seed cho LLM client (OpenAI, NVIDIA endpoints).
- Seed được tự động ghi nhận trong `manifest.json`, `report.json` và bảng báo cáo `report.md`.

### 2.6. Hệ thống Đánh giá & Metric (`src/evaluator/`)
Đánh giá song song 2 khía cạnh:
1. **Black-box Functional Correctness (Chấm trên Test Set ẩn):**
   - **`Benchmark Transformed Accuracy` (Metric chính):** Độ chính xác trên tập các cột **thực sự bị biến đổi** (các task timeout/quá budget bị phạt điểm `0.0`).
   - **`Functional Column Accuracy`:** Độ chính xác trên toàn bộ cột của bảng.
2. **White-box Symbolic Recovery (Chấm trên biểu thức Operator):**
   - **`Operator F1-Score`:** Đo độ chính xác của tập toán tử suy luận so với ground truth (hỗ trợ alias normalization như `np.log` $\to$ `LogTransformation`).
   - **`Exact Sequence Accuracy`:** Khôi phục chính xác 100% thứ tự chuỗi phép toán.
3. **Báo cáo phân tầng:**
   - **`Benchmark Scores (All Tasks)`:** Điểm tổng kết toàn Suite (task fail = 0 điểm).
   - **`Conditional Metrics (Submitted Only)`:** Điểm khảo sát chất lượng khi mô hình nộp bài thành công.
   - **`Submission Rate`:** Tỉ lệ hoàn thành task trong ngân sách.

---

## 3. Cấu trúc Thư mục Kết quả Pipeline (`pipeline_runs/<timestamp>/`)

Mỗi lượt chạy sinh ra một thư mục cô lập gồm:
```text
pipeline_runs/<timestamp>/
├── manifest.json       # Metadata tái lập thí nghiệm (seed, dataset, model, config, task specs)
├── tasks/              # Các task đã sinh (train_before, train_after, hidden/test_*)
├── logs/               # Log chi tiết JSONL từng lượt gọi LLM, mã python và stdout
├── artifacts/          # Script transform(df) độc lập được agent nộp
├── results/            # Kết quả JSON chi tiết từng task
└── reports/
    ├── summary.csv     # Bảng tổng hợp phẳng toàn bộ các run
    ├── report.json     # Báo cáo dạng JSON cho máy đọc
    └── report.md       # Báo cáo Markdown trực quan (Metadata, Seed, Benchmark & Conditional scores)
```

