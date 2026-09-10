# Tabular Transformation Discovery Harness

Research harness hai tầng để đo khả năng LLM khám phá quy luật biến đổi theo cột.

- **Tier 1 — discovery:** LangGraph cho model thấy duy nhất `before.csv`, `after.csv` và hai tool `run_python(code)`, `submit_solution(script, transformations)`.
- **Tier 2 — evaluation:** chạy `transform(df)` trong một sandbox mới trên hidden input. Hidden expected output không được mount vào sandbox và không bao giờ được trả lại cho agent.

Harness áp hard budget ở controller cho input/output token, tổng tool call, execution wall time, CPU time và toàn episode wall time. Mỗi run sinh trajectory JSONL, `transform.py` độc lập và result JSON phù hợp để tổng hợp số liệu nghiên cứu.

> [!NOTE]
> Tài liệu chi tiết về kiến trúc hệ thống, quy trình thực thi, sandbox và công thức metric xem tại: [docs/harness_architecture.md](docs/harness_architecture.md).

## Cài đặt

```powershell
Copy-Item .env.example .env
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

Điền key vào `.env`. Không commit file này. NVIDIA và OpenAI được hỗ trợ sẵn; provider khác có thể đăng ký bằng `register_provider()` trong `src/model_factory.py` mà không đổi graph.

```dotenv
BENCHMARK_MODEL_PROVIDER=nvidia
BENCHMARK_MODEL=nvidia/nemotron-3-super-120b-a12b
BENCHMARK_API_KEY_ENV=NVIDIA_API_KEY
NVIDIA_API_KEY=...
```

Các biến `BENCHMARK_*` override file model YAML. File mẫu nằm ở `configs/nvidia.yaml` và `configs/openai.yaml`.

## Chạy toàn bộ pipeline end-to-end

Với dataset và config mặc định của project, chỉ cần điền API key trong `.env` rồi chạy:

```powershell
python run_pipeline.py
```

Mặc định script dùng `notebooks/processed_data.csv`, `configs/nvidia.yaml`, `configs/harness.yaml`, sinh 24 task, mỗi task sample 10.000 dòng và dùng seed 42. Các CLI option vẫn có thể dùng khi cần override.

Mỗi lần chạy tạo một thư mục cô lập trong `pipeline_runs/<UTC timestamp>/` gồm task đã sinh, trajectory, `transform.py`, raw result JSON, `summary.csv`, `report.json`, `report.md` và manifest tái lập thí nghiệm. Mặc định mỗi task sample xác định 10.000 dòng để tránh nhân bản toàn bộ dataset lớn; dùng `--rows-per-task 0` nếu muốn dùng tất cả. Dùng `--run-name <name>` nếu muốn tên cố định.

Có thể kiểm tra dữ liệu và sinh task mà chưa cần API key:

```powershell
.venv\Scripts\python.exe run_pipeline.py `
  --source notebooks\processed_data.csv `
  --model-config configs\nvidia.yaml `
  --prepare-only
```

## Chạy một task

Task cũ có cấu trúc:

```text
data/0/
  train_before.csv
  train_after.csv
  config.yaml
```

Chạy discovery (nếu không có hidden pair thì evaluation được bỏ qua):

```powershell
.venv\Scripts\python.exe -m src.benchmark.runner `
  --task data\0 `
  --model-config configs\nvidia.yaml `
  --harness-config configs\harness.yaml
```

Gắn hidden pair nằm ngoài task:

```powershell
.venv\Scripts\python.exe -m src.benchmark.runner `
  --task data\0 `
  --model-config configs\nvidia.yaml `
  --test-before path\to\test_before.csv `
  --test-after path\to\test_after.csv
```

Task do builder mới tạo có `hidden/test_before.csv` và `hidden/test_after.csv`; loader tự gắn chúng. Discovery vẫn chỉ mount hai file train, evaluator chỉ mount hidden input.

Đầu ra mỗi run:

```text
logs/<task>-<run_id>.jsonl       # response, tool code/result, usage, submission, metrics
artifacts/<run_id>/transform.py  # self-contained transform(df)
results/<run_id>.json            # budget, status, functional/operator metrics
```

Metric gồm functional column accuracy, chi tiết đúng/sai từng cột, operator precision/recall/F1 và exact ordered operator-sequence accuracy.

Gộp mọi result JSON thành một bảng CSV phẳng để phân tích:

```powershell
.venv\Scripts\python.exe -m src.benchmark.aggregate --results results --output results\summary.csv
```

Chạy tuần tự toàn bộ task con và tự tạo summary:

```powershell
.venv\Scripts\python.exe -m src.benchmark.suite `
  --task-root generated_tasks `
  --model-config configs\nvidia.yaml `
  --harness-config configs\harness.yaml
```

## Sinh task theo cấp độ khó (Difficulty Levels)

Harness hỗ trợ sinh task đa tầng theo độ sâu cây cú pháp AST (Level 0 → Level 5):
- **Level 0 (Atomic):** 1 phép biến đổi đơn biến $f(x)$ ($d = 1$).
- **Level 1 (Atomic Unary):** 2 phép biến đổi đơn biến độc lập ($d = 1$, không lồng ghép).
- **Level 2 (Composed Unary):** 3 phép biến đổi lồng ghép $f(g(x))$ ($d = 2$, bảo đảm lồng ghép chuỗi hàm).
- **Level 3 (Interactions):** 3 cột độc lập + 1 tương tác phụ thuộc đa cột ($C = A \text{ op } B$, $d = 2$).
- **Level 4 (Conditional / If-Else):** 3 cột độc lập (chứa hàm phân mảnh / indicator) + 1 tương tác điều kiện phân mảnh.
- **Level 5 (Hybrid Deep AST):** 4 cột độc lập (lồng ghép + conditional) + 2 tương tác đa biến với độ sâu $d \ge 3$.

Tập 24 phép biến đổi chuẩn hóa:
- **Đơn biến & Phi tuyến:** `LogTransformation`, `Log1pTransformation`, `SquareRootTransformation`, `SquareTransformation`, `CubeTransformation`, `CubeRootTransformation`, `ExpTransformation`, `InverseTransformation`, `AbsTransformation`.
- **Chuẩn hóa:** `StandardScalerTransformation`, `MinMaxScalerTransformation`.
- **Lượng giác & Kích hoạt:** `SinTransformation`, `CosTransformation`, `TanhTransformation`, `SigmoidTransformation`.
- **Phân mảnh & Điều kiện (Piecewise / Conditional):** `IndicatorZeroTransformation` ($\mathbb{I}(x > 0)$), `IndicatorMeanTransformation` ($\mathbb{I}(x > \text{mean}(x))$), `ClipZeroTransformation` ($\max(x, 0)$ / ReLU), `PiecewiseLogTransformation` ($\log(x) \text{ if } x > 1 \text{ else } x$).
- **Tương tác đa cột (Inter-column):** `AddInteractionTransformation` ($A + B$), `SubtractInteractionTransformation` ($A - B$), `MultiplyInteractionTransformation` ($A \times B$), `DivideInteractionTransformation` ($A / B$), `ConditionalInteractionTransformation` ($A \text{ if } A > \text{mean}(A) \text{ else } B$).

Harness tự động áp dụng **Ngân sách thích ứng (Tiered Budget)** theo từng Level (Level 1: 40k tokens $\to$ Level 5: 120k tokens), đồng thời cố định `--seed 42` để đảm bảo tính tái lập 100% trên model và task generation.

Sinh task qua CLI `run_pipeline.py` hoặc `task_builder`:

```powershell
# Chạy pipeline với Level 3 và seed 42
.venv\Scripts\python.exe run_pipeline.py --level 3 --task-count 24 --seed 42

# Hoặc chỉ sinh task kiểm tra trước
.venv\Scripts\python.exe -m src.benchmark.task_builder `
  --source notebooks\processed_data.csv `
  --output generated_tasks\0 `
  --level 3 `
  --seed 42
```


## Sandbox

`local` là backend mặc định để phát triển/test. Nó chạy code trong child process cô lập, scrub environment (không truyền API key), giới hạn timeout/output, nhưng **không phải security boundary**.

Để dùng Docker, chỉnh `configs/harness.yaml`:

```yaml
sandbox:
  backend: docker
  docker_image: tabular-benchmark-sandbox:latest
  docker_network: none
  docker_cpus: 1
  docker_memory: 1g
  docker_pids_limit: 64
```

Hoặc override bằng `--sandbox-backend docker --docker-image <image>`. Image do người dùng quản lý; contract tối thiểu là có `python`, `pandas`, `numpy`, shell và `tail`. Adapter tạo container mới cho mỗi discovery/evaluation sandbox, bind đúng workspace tạm, mặc định tắt network, rồi xóa container/workspace khi hoàn tất. Không mount Docker socket, home hoặc secrets.

## Cấu hình benchmark

`configs/harness.yaml` chứa toàn bộ budget, tolerance, operator vocabulary và Docker knobs. Model YAML chỉ chứa model/provider/key-env. API key chỉ được model client đọc ở host; generated code không nhận environment của host.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Test suite không cần API key hay Docker. Nó kiểm tra graph run→run→submit, mọi hard budget, submission AST validation, clean evaluator boundary, local subprocess execution, operator metrics và applicability/task generation.
