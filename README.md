# Heart_Module

Professional multi-organ clinical dashboard with doctor login and analysis modules for:
- Heart reconstruction and progression
- Brain detection and progression
- Lung detection
- Liver detection
- Unified all-in-one upload flow with disease selector
- Separate progression workflow with disease selector
- Patient name registry saved on each upload/progression run

## Doctor Login
- Username: `doctor`
- Password: `doctor123`
- Multi-page flow:
  - `login.html` (login/register gateway)
  - `dashboard.html` (clinical app dashboard)
- Login mode asks only: username + password
- Register mode asks: doctor name, staff ID, hospital name, email, username, password
- Register creates local doctor account in `uploads/doctor_registry.json`

## Home Page Flow
- Single analysis form: patient name + disease type + files
- Separate progression form: patient name + progression disease type + T1/T2 files
- Profile card
- Recent patient history panel (saved in `uploads/patient_registry.json`)
- Full-screen 2D-to-3D reconstruction viewport
- Email export action for latest generated report

## Best Lung/Liver Model Workflow
The app automatically searches and uses the strongest available custom checkpoints first:
- `models/lung_best_model.keras`
- `models/liver_best_model.keras`

If these are not present, it falls back to transfer-based pretrained inference and finally heuristic logic.

## Train Your Own Best Models
Expected dataset layout:

`data/lung/normal`, `data/lung/abnormal`  
`data/liver/normal`, `data/liver/abnormal`

Train commands:

```bash
python models/train_lung_model.py
python models/train_liver_model.py
```

These scripts use transfer learning, validation AUC checkpointing, early stopping, and fine-tuning.

## Model Choice Rationale (from literature trend)
- `EfficientNet` transfer learning is selected as the default practical backbone for lung/liver slice classification due to strong accuracy/efficiency tradeoff.
- For full 3D segmentation-heavy pipelines, `nnU-Net` is a strong benchmark family and can be added in future if you want dedicated organ segmentation training.

## Email Export Configuration (Optional SMTP)
Set these environment variables for direct SMTP sending:

- `SMTP_HOST`
- `SMTP_PORT` (default `587`)
- `SMTP_USER`
- `SMTP_PASS`
- `SMTP_SENDER` (optional, defaults to SMTP_USER)

If SMTP is not configured, the app falls back to opening a `mailto:` draft.
