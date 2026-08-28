# Tabular Transformation Discovery Harness

Research harness hai tầng để đo khả năng LLM khám phá quy luật biến đổi theo cột.

- **Tier 1 — discovery:** LangGraph cho model thấy duy nhất `before.csv`, `after.csv` và hai tool `run_python(code)`, `submit_solution(script, transformations)`.
- **Tier 2 — evaluation:** chạy `transform(df)` trong một sandbox mới trên hidden input. Hidden expected output không được mount vào sandbox và không bao giờ được trả lại cho agent.

Harness áp hard budget ở controller cho input/output token, tổng tool call, execution wall time, CPU time và toàn episode wall time. Mỗi run sinh trajectory JSONL, `transform.py` độc lập và result JSON phù hợp để tổng hợp số liệu nghiên cứu.

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

## Sinh task atomic hợp lệ

Builder chỉ nhận dữ liệu numeric, split train/hidden theo seed, rồi chọn trong các cặp cột–operator đã qua kiểm tra applicability trên toàn nguồn. Vì vậy `LogTransformation` không được áp vào giá trị `<= 0`, `SquareRootTransformation` không áp vào giá trị âm, và mọi output NaN/inf đều bị loại.

```powershell
.venv\Scripts\python.exe -m src.benchmark.task_builder `
  --source notebooks\processed_data.csv `
  --output generated_tasks\0 `
  --seed 42 `
  --operators LogTransformation SquareRootTransformation SquareTransformation
```

Khi thêm operator mới, đăng ký class trong `OPERATORS` của `src/benchmark/task_builder.py` và implement `is_applicable`; implementation mặc định cũng từ chối exception hoặc output không hữu hạn.

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
