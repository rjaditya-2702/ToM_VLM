# BENCHMARKING MODELS FOR DESIRE

## Complete File Tree

```
vqa_benchmark/
│
│
├── data/
│   ├── datafactory.py          # To format the prompt into appropriate model's chat template
│   ├── prepare_experiment.py   # DataLoaderFactory class
│   ├── sample.csv              # Your dataset (YOU CREATE)
│   └── __init__.py
│
├── models/
│   ├── model_factory.py        # ModelFactory (creates model instances)
│   ├── llama.py                # Model specific code to infer
│   ├── llava7b.py              
│   ├── llava13b.py             
│   ├── gemma.py                
│   ├── qwen.py                 
│   ├── bagel.py                
│   ├── __init__.py
|
├── runner.py                    # BenchmarkRunner (orchestrator)
│
├── results/                     # Auto-created
│   └── *.csv                    # Aggregated results
│
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│                     (CLI Interface)                         │
│  - Parses arguments (model, experiment, batch_size)         │
│  - Creates BenchmarkRunner instance                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  BenchmarkRunner                            │
│                     (runner.py)                             │
│  - Orchestrates the entire benchmark process                │
│  - Coordinates: Factory → Dataloader → Model → Results      │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│ModelFactory │   │DataLoader    │   │ ResultsWriter   │
│             │   │Factory       │   │                 │
│             │   │              │   │writes:          │
│             │   |              │   │all_results.csv  │
│             │   |              │   │                 │
│creates →    │   │              │   │                 │
│Model        │   │              │   │                 │
│Instance     │   │creates →     │   │                 │
│             │   │DataLoader    │   │                 │
└─────────────┘   └──────────────┘   └─────────────────┘

```

## Quick Start
```
# 1. Install dependencies  
pip install -r requirements.txt

# 2. Create your data/sample.csv with columns:
#    image_name, caption, tru_desire

# 3. Run a benchmark
python main.py --model qwen --experiment expt4 --batch-size 32

# 4. View results
cat results/qwen_expt4.csv
```

## Execution Flow (Step by Step)

```
1. User runs: python main.py --model gemma --experiment expt4 --batch-size 32
   ↓
2. main.py creates BenchmarkRunner
   ↓
3. run_benchmark() is called
   ↓
6. DataFactory.create_model("expt4", 32) creates DataLoader
   │  - For expt4: uses real images + questions
   │  - Creates batches of size 32
   ↓
4. ModelFactory.create_model("gemma") creates gemma instance
   ↓
7. Model.infer(dataloader) processes all batches
   │  - Loops through dataloader internally
   │  - Runs inference on each batch
   │  - Collects predictions (text, option token probabilities)
   │  - Returns a pandas df
   ↓
8. save_results() writes to CSV
   │  - Adds model_name and experiment_type columns
   │  - Appends to results/<model_name>_<expt_no>.csv
   ↓
10. Done! Results saved.
```

## Key Design Patterns

### 1. Factory Pattern (Model Creation)
```python
# Instead of:
if model_name == "Qwen":
    model = Qwen()
elif model_name == "llama":
    model = Llama()
# ...more if/else

# We use:
model = ModelFactory.get_model(model_name)  # Dynamic!
```

### 2. Abstract Base Class (Consistency)
```python
# Every model MUST implement these:
class YourModel(AbstractVQAModel):
    def load(self, device="cuda"):
        # Load model weights
        
    def infer(self, dataloader):
        # Run inference, return results
```

### 3. Strategy Pattern (Experiments)
```python
# Same dataset, different preprocessing:
if experiment == "expt1": use blank image, no question
if experiment == "expt2": use blank image, with question
if experiment == "expt3": use real image, no question
if experiment == "expt4": use real image, with question
```

### 4. Separation of Concerns
```
Data Preparation  →  Models don't know about experiments
Model Logic       →  Doesn't know about data format
Results Writing   →  Doesn't know about models
Orchestration     →  Coordinates everything
```

## Implementation Checklist

### Initial Setup
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create your `data/msed_dev.csv` with proper format

### Add Your First Model
- [ ] Create file: `models/implementations/your_model.py`
- [ ] Inherit from `AbstractVQAModel`
- [ ] Implement `load()` method
- [ ] Implement `infer(dataloader)` method
- [ ] Add entry to `config/models.yaml`
- [ ] Test: `python main.py --model your_model --experiment expt4 --batch-size 8`

### Run Benchmarks
- [ ] Test single run: `python main.py --model llava7b --experiment expt4 --batch-size 32`
- [ ] Run all experiments for one model (loop or parallel)
- [ ] Check results: `results/all_results.csv`
- [ ] Compute metrics on results

## Customization Points

### 1. Add Image Transforms
In `data/base_dataset.py`:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = VQADataset(..., transform=transform)
```

### 2. Custom Experiments
Add to `config/experiments.yaml`:
```yaml
expt5:
  name: "Custom Experiment"
  description: "Your custom setup"
```

Then modify `VQADataset.__getitem__()` to handle it.

### 3. Different Result Formats
In `inference/results_writer.py`, modify `save_results()` to:
- Save to JSON instead of CSV
- Save to database
- Upload to cloud storage
- Generate visualizations

### 4. Multi-GPU Support
In your model's `load()` method:
```python
def load(self, device="cuda"):
    self.model = nn.DataParallel(self.model)
    self.model.to(device)
```

## Understanding the Code

### Why separate DataLoader per experiment?
- Each experiment needs different data (blank vs real images, with/without questions)
- Cleaner than having complex logic inside model
- Easy to add new experiment types

### Why pass entire DataLoader to model.infer()?
- Model controls batching loop
- Can implement model-specific optimizations
- Consistent interface across all models

### Why use Factory pattern?
- Add new models without changing existing code
- Config-driven (just update YAML)
- No if/else chains
- Easy to extend

### Why Abstract Base Class?
- Enforces consistency across all models
- Type safety
- Clear contract: "every model must implement load() and infer()"
- Makes adding new models foolproof

## Next Steps

1. **Add more models**: Implement different architectures
2. **Compute metrics**: Add accuracy, BLEU, CIDEr calculations
3. **Visualizations**: Plot performance comparisons
4. **Ablation studies**: Test different model components
5. **Error analysis**: Examine failure cases
6. **Optimization**: Profile and speed up inference
