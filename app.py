import json
import os
import smtplib
from datetime import datetime
from functools import wraps
from email.message import EmailMessage
from urllib.parse import quote

from flask import Flask, make_response, redirect, render_template, request, session, url_for
import plotly.graph_objects as go

from models.brain_model import predict_brain_tumor
from models.liver_model import predict_liver_condition
from models.lung_model import predict_lung_condition
from preprocessing.scan_validation import validate_scan_folder
from reconstruction.brain_3d import generate_3d_brain
from reconstruction.brain_progression import compare_brain_progression
from reconstruction.heart_3d import generate_3d_heart
from reconstruction.progression import compare_progression

APP_NAME = "Multi-Organ Imaging Workbench"
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

DOCTOR_USERNAME = os.getenv("DOCTOR_USERNAME", "doctor")
DOCTOR_PASSWORD = os.getenv("DOCTOR_PASSWORD", "doctor123")

UPLOAD_FOLDER = "uploads"
PATIENTS_FILE = os.path.join(UPLOAD_FOLDER, "patient_registry.json")
DOCTORS_FILE = os.path.join(UPLOAD_FOLDER, "doctor_registry.json")

HEART_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "heart")
BRAIN_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "brain")
LUNG_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "lung")
LIVER_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "liver")

HEART_T1_FOLDER = os.path.join(UPLOAD_FOLDER, "heart_t1")
HEART_T2_FOLDER = os.path.join(UPLOAD_FOLDER, "heart_t2")
BRAIN_T1_FOLDER = os.path.join(UPLOAD_FOLDER, "brain_t1")
BRAIN_T2_FOLDER = os.path.join(UPLOAD_FOLDER, "brain_t2")
LUNG_T1_FOLDER = os.path.join(UPLOAD_FOLDER, "lung_t1")
LUNG_T2_FOLDER = os.path.join(UPLOAD_FOLDER, "lung_t2")
LIVER_T1_FOLDER = os.path.join(UPLOAD_FOLDER, "liver_t1")
LIVER_T2_FOLDER = os.path.join(UPLOAD_FOLDER, "liver_t2")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def _load_patient_registry():
    if not os.path.exists(PATIENTS_FILE):
        return []
    try:
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _load_doctor_registry():
    if not os.path.exists(DOCTORS_FILE):
        return []
    try:
        with open(DOCTORS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_doctor_registry(entries):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with open(DOCTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _save_patient_registry(entries):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _register_patient(patient_name, disease_type, stage, clinical_status="recovering"):
    name = (patient_name or "").strip()
    if not name:
        return
    doctor = session.get("doctor_profile", {})
    entries = _load_patient_registry()
    entries.append(
        {
            "patient_name": name,
            "disease_type": disease_type,
            "stage": stage,
            "clinical_status": (clinical_status or "recovering").lower(),
            "doctor_username": doctor.get("username", ""),
            "doctor_name": doctor.get("name", ""),
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        }
    )
    _save_patient_registry(entries[-200:])


def _recent_patients(limit=10):
    entries = _load_patient_registry()
    return list(reversed(entries[-limit:]))


def _profile_snapshot():
    doctor = session.get("doctor_profile", {})
    username = doctor.get("username", "")
    entries = [e for e in _load_patient_registry() if e.get("doctor_username") == username]

    disease_counts = {}
    unique_patients = set()
    recovering = {}
    recovered = {}

    for entry in entries:
        p = entry.get("patient_name", "Unknown")
        d = entry.get("disease_type", "unknown")
        s = entry.get("clinical_status", "recovering")
        unique_patients.add(p)
        disease_counts[d] = disease_counts.get(d, 0) + 1
        if s == "recovered":
            recovered[p] = entry
            if p in recovering:
                recovering.pop(p, None)
        else:
            if p not in recovered:
                recovering[p] = entry

    case_studies = list(reversed(entries[-12:]))
    return {
        "total_cases": len(entries),
        "total_patients": len(unique_patients),
        "disease_counts": disease_counts,
        "recovering_patients": list(recovering.keys()),
        "recovered_patients": list(recovered.keys()),
        "case_studies": case_studies,
    }


def _set_last_report(patient_name, disease_type, stage, result):
    result_pairs = []
    for key, value in result.items():
        if key.endswith("_html"):
            continue
        result_pairs.append(f"{key}: {value}")
    session["last_report"] = {
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "patient_name": patient_name or "Unknown",
        "disease_type": disease_type,
        "stage": stage,
        "result_lines": result_pairs,
    }


def _build_report_text():
    report = session.get("last_report")
    if not report:
        return "No report found in current session."
    lines = [
        f"Clinical Insight Report",
        f"Timestamp: {report.get('timestamp', 'N/A')}",
        f"Patient: {report.get('patient_name', 'Unknown')}",
        f"Disease Type: {report.get('disease_type', 'Unknown')}",
        f"Stage: {report.get('stage', 'Unknown')}",
        "",
        "Results:",
    ]
    lines.extend(report.get("result_lines", []))
    return "\n".join(lines)


def render_home(**kwargs):
    is_authenticated = bool(session.get("doctor_authenticated"))
    profile = session.get("doctor_profile", {})
    template_name = "dashboard.html" if is_authenticated else "login.html"
    return render_template(
        template_name,
        app_name=APP_NAME,
        is_authenticated=is_authenticated,
        doctor_profile=profile,
        patient_history=_recent_patients(),
        **kwargs,
    )


def doctor_required(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        if not session.get("doctor_authenticated"):
            return render_home(login_error="Doctor authentication required.")
        return route_func(*args, **kwargs)

    return wrapper


def _save_files(files, target_folder):
    clear_folder(target_folder)
    saved_any = False
    for file in files:
        if file and file.filename:
            file.save(os.path.join(target_folder, file.filename))
            saved_any = True
    return saved_any


def _heart_analysis(folder):
    is_valid, validation_msg = validate_scan_folder(folder, "heart")
    if not is_valid:
        raise ValueError(validation_msg)

    fig, tumor_percentage, severity, lesion_centroid = generate_3d_heart(folder)
    return {
        "heart_graph_html": fig.to_html(full_html=False),
        "heart_tumor_percentage": round(tumor_percentage, 2),
        "heart_severity": severity,
        "heart_lesion_centroid": lesion_centroid,
    }


def _brain_analysis(folder):
    is_valid, validation_msg = validate_scan_folder(folder, "brain")
    if not is_valid:
        raise ValueError(validation_msg)

    fig, tumor_burden, tumor_centroid = generate_3d_brain(folder)
    prediction_label, confidence_percent, prediction_method = predict_brain_tumor(folder)
    if prediction_label == "Tumor Detected":
        brain_risk_score = float(confidence_percent)
    else:
        brain_risk_score = float(max(0.0, 100.0 - confidence_percent))
    return {
        "brain_graph_html": fig.to_html(full_html=False),
        "brain_prediction": prediction_label,
        "brain_confidence": round(confidence_percent, 2),
        "brain_model_method": prediction_method,
        "brain_tumor_burden": round(tumor_burden, 2),
        "brain_tumor_centroid": tumor_centroid,
        "brain_risk_score": round(brain_risk_score, 2),
    }


def _lung_analysis(folder):
    prediction, confidence, method, risk_score = predict_lung_condition(folder)
    return {
        "lung_prediction": prediction,
        "lung_confidence": round(confidence, 2),
        "lung_model_method": method,
        "lung_risk_score": round(risk_score, 2),
    }


def _liver_analysis(folder):
    prediction, confidence, method, risk_score = predict_liver_condition(folder)
    return {
        "liver_prediction": prediction,
        "liver_confidence": round(confidence, 2),
        "liver_model_method": method,
        "liver_risk_score": round(risk_score, 2),
    }


def _heart_progression():
    ok_t1, msg_t1 = validate_scan_folder(HEART_T1_FOLDER, "heart")
    ok_t2, msg_t2 = validate_scan_folder(HEART_T2_FOLDER, "heart")
    if not ok_t1 or not ok_t2:
        raise ValueError(msg_t1 or msg_t2)

    fig, metrics = compare_progression(HEART_T1_FOLDER, HEART_T2_FOLDER)
    return {
        "heart_progression_graph_html": fig.to_html(full_html=False),
        "heart_t1_burden": round(metrics["t1_burden"], 2),
        "heart_t2_burden": round(metrics["t2_burden"], 2),
        "heart_absolute_growth": round(metrics["absolute_change"], 2),
        "heart_relative_growth": round(metrics["relative_change"], 2),
        "heart_growth_voxels": metrics["growth_voxels"],
        "heart_regression_voxels": metrics["regression_voxels"],
    }


def _simple_progression_plot(organ_label, t1_score, t2_score):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=["T1", "T2"],
            y=[t1_score, t2_score],
            mode="lines+markers",
            line=dict(color="#2e8bff", width=4),
            marker=dict(size=10),
            name=f"{organ_label} Risk",
        )
    )
    fig.update_layout(
        title=f"{organ_label} Progression Risk Trend",
        yaxis_title="Risk Score (0-100)",
        xaxis_title="Timepoint",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _brain_progression(interval_value, interval_unit):
    ok_t1, msg_t1 = validate_scan_folder(BRAIN_T1_FOLDER, "brain")
    ok_t2, msg_t2 = validate_scan_folder(BRAIN_T2_FOLDER, "brain")
    if not ok_t1 or not ok_t2:
        raise ValueError(msg_t1 or msg_t2)

    fig, metrics = compare_brain_progression(BRAIN_T1_FOLDER, BRAIN_T2_FOLDER)
    period_label = f"{interval_value} {interval_unit}{'' if interval_value == 1 else 's'}"
    monthly_scale = interval_value if interval_unit == "month" else interval_value / 4.345
    monthly_scale = monthly_scale if monthly_scale > 0 else 1.0
    normalized_monthly_growth = metrics["absolute_growth"] / monthly_scale

    return {
        "brain_progression_graph_html": fig.to_html(full_html=False),
        "brain_t1_burden": round(metrics["t1_burden"], 2),
        "brain_t2_burden": round(metrics["t2_burden"], 2),
        "brain_absolute_growth": round(metrics["absolute_growth"], 2),
        "brain_relative_growth": round(metrics["relative_growth"], 2),
        "brain_growth_voxels": metrics["growth_voxels"],
        "brain_regression_voxels": metrics["regression_voxels"],
        "progression_period_label": period_label,
        "brain_monthly_growth": round(normalized_monthly_growth, 2),
    }


def _lung_progression():
    _, _, _, t1_risk = predict_lung_condition(LUNG_T1_FOLDER)
    _, _, _, t2_risk = predict_lung_condition(LUNG_T2_FOLDER)
    absolute_change = float(t2_risk - t1_risk)
    relative_change = float((absolute_change / max(1e-6, t1_risk)) * 100.0) if t1_risk > 0 else 0.0
    fig = _simple_progression_plot("Lung", t1_risk, t2_risk)
    return {
        "lung_progression_graph_html": fig.to_html(full_html=False),
        "lung_t1_risk": round(t1_risk, 2),
        "lung_t2_risk": round(t2_risk, 2),
        "lung_absolute_growth": round(absolute_change, 2),
        "lung_relative_growth": round(relative_change, 2),
    }


def _liver_progression():
    _, _, _, t1_risk = predict_liver_condition(LIVER_T1_FOLDER)
    _, _, _, t2_risk = predict_liver_condition(LIVER_T2_FOLDER)
    absolute_change = float(t2_risk - t1_risk)
    relative_change = float((absolute_change / max(1e-6, t1_risk)) * 100.0) if t1_risk > 0 else 0.0
    fig = _simple_progression_plot("Liver", t1_risk, t2_risk)
    return {
        "liver_progression_graph_html": fig.to_html(full_html=False),
        "liver_t1_risk": round(t1_risk, 2),
        "liver_t2_risk": round(t2_risk, 2),
        "liver_absolute_growth": round(absolute_change, 2),
        "liver_relative_growth": round(relative_change, 2),
    }


@app.route("/")
def index():
    return render_home()


@app.route("/profile")
@doctor_required
def profile():
    return render_template(
        "profile.html",
        app_name=APP_NAME,
        doctor_profile=session.get("doctor_profile", {}),
        profile_data=_profile_snapshot(),
    )


@app.route("/doctor/login", methods=["POST"])
def doctor_login():
    auth_mode = request.form.get("auth_mode", "login").strip().lower()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    doctor_name = request.form.get("doctor_name", "").strip()
    staff_id = request.form.get("staff_id", "").strip()
    hospital_name = request.form.get("hospital_name", "").strip()
    doctor_email = request.form.get("doctor_email", "").strip()

    if not username or not password:
        return render_home(login_error="Username and password are required.")

    doctors = _load_doctor_registry()
    doctor_match = next((d for d in doctors if d.get("username") == username), None)

    if auth_mode == "register":
        if not doctor_name or not staff_id or not hospital_name or not doctor_email:
            return render_home(
                login_error="For registration, name, staff ID, hospital name, and email are required."
            )
        if doctor_match is not None:
            return render_home(login_error="Username already registered. Use login mode.")
        doctors.append(
            {
                "username": username,
                "password": password,
                "name": doctor_name,
                "staff_id": staff_id,
                "hospital_name": hospital_name,
                "doctor_email": doctor_email,
                "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            }
        )
        _save_doctor_registry(doctors)
    else:
        default_ok = username == DOCTOR_USERNAME and password == DOCTOR_PASSWORD
        registry_ok = doctor_match is not None and doctor_match.get("password") == password
        if not (default_ok or registry_ok):
            return render_home(login_error="Invalid credentials. Use Register mode if new.")
        if registry_ok:
            doctor_name = doctor_match.get("name", doctor_name)
            staff_id = doctor_match.get("staff_id", staff_id)
            hospital_name = doctor_match.get("hospital_name", hospital_name)
            doctor_email = doctor_match.get("doctor_email", doctor_email)
        else:
            doctor_name = doctor_name or f"Dr. {username.title()}"
            staff_id = staff_id or "NA"
            hospital_name = hospital_name or "Clinical Network"

    session["doctor_authenticated"] = True
    session["doctor_profile"] = {
        "name": doctor_name,
        "username": username,
        "staff_id": staff_id,
        "department": "Diagnostic Imaging",
        "hospital": hospital_name,
        "email": doctor_email,
        "last_login": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    return redirect(url_for("index"))


@app.route("/doctor/logout", methods=["POST"])
def doctor_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
@doctor_required
def analyze():
    patient_name = request.form.get("patient_name", "").strip()
    disease_type = request.form.get("disease_type", "").strip().lower()
    clinical_status = request.form.get("clinical_status", "recovering").strip().lower()
    files = request.files.getlist("disease_files")

    disease_folder_map = {
        "heart": HEART_UPLOAD_FOLDER,
        "brain": BRAIN_UPLOAD_FOLDER,
        "lung": LUNG_UPLOAD_FOLDER,
        "liver": LIVER_UPLOAD_FOLDER,
    }

    if disease_type not in disease_folder_map:
        return render_home(result_error="Select a valid disease type.", selected_disease="heart")

    if not _save_files(files, disease_folder_map[disease_type]):
        return render_home(
            result_error="Upload one or more files before running analysis.",
            selected_disease=disease_type,
        )

    if clinical_status not in ("recovering", "recovered"):
        clinical_status = "recovering"
    _register_patient(patient_name, disease_type, "analysis", clinical_status)

    try:
        if disease_type == "heart":
            result = _heart_analysis(disease_folder_map[disease_type])
        elif disease_type == "brain":
            result = _brain_analysis(disease_folder_map[disease_type])
        elif disease_type == "lung":
            result = _lung_analysis(disease_folder_map[disease_type])
        else:
            result = _liver_analysis(disease_folder_map[disease_type])
    except ValueError as exc:
        return render_home(
            result_error=str(exc),
            selected_disease=disease_type,
        )

    _set_last_report(patient_name, disease_type, "analysis", result)
    return render_home(
        selected_disease=disease_type,
        **result,
    )


@app.route("/progression/run", methods=["POST"])
@doctor_required
def progression_run():
    patient_name = request.form.get("progression_patient_name", "").strip()
    progression_disease = request.form.get("progression_disease_type", "heart").strip().lower()
    clinical_status = request.form.get("progression_status", "recovering").strip().lower()
    t1_files = request.files.getlist("t1_files")
    t2_files = request.files.getlist("t2_files")

    if progression_disease not in ("heart", "brain", "lung", "liver"):
        return render_home(
            progression_error="Progression supports heart, brain, lung, and liver.",
            progression_selected_disease="heart",
        )

    if progression_disease == "heart":
        t1_ok = _save_files(t1_files, HEART_T1_FOLDER)
        t2_ok = _save_files(t2_files, HEART_T2_FOLDER)
    elif progression_disease == "brain":
        t1_ok = _save_files(t1_files, BRAIN_T1_FOLDER)
        t2_ok = _save_files(t2_files, BRAIN_T2_FOLDER)
    elif progression_disease == "lung":
        t1_ok = _save_files(t1_files, LUNG_T1_FOLDER)
        t2_ok = _save_files(t2_files, LUNG_T2_FOLDER)
    else:
        t1_ok = _save_files(t1_files, LIVER_T1_FOLDER)
        t2_ok = _save_files(t2_files, LIVER_T2_FOLDER)

    if not t1_ok or not t2_ok:
        return render_home(
            progression_error="Upload both T1 and T2 scan files.",
            progression_selected_disease=progression_disease,
        )

    if clinical_status not in ("recovering", "recovered"):
        clinical_status = "recovering"
    _register_patient(patient_name, progression_disease, "progression", clinical_status)

    try:
        if progression_disease == "heart":
            result = _heart_progression()
        elif progression_disease == "brain":
            interval_value_text = request.form.get("interval_value", "1").strip()
            interval_unit = request.form.get("interval_unit", "week").strip().lower()
            try:
                interval_value = max(1, int(interval_value_text))
            except ValueError:
                interval_value = 1
            if interval_unit not in ("week", "month"):
                interval_unit = "week"
            result = _brain_progression(interval_value, interval_unit)
        elif progression_disease == "lung":
            result = _lung_progression()
        else:
            result = _liver_progression()
    except ValueError as exc:
        return render_home(
            progression_error=str(exc),
            progression_selected_disease=progression_disease,
        )

    _set_last_report(patient_name, progression_disease, "progression", result)
    return render_home(
        progression_selected_disease=progression_disease,
        **result,
    )


@app.route("/report/export_email", methods=["POST"])
@doctor_required
def export_email():
    recipient_email = request.form.get("recipient_email", "").strip()
    if not recipient_email:
        return render_home(export_error="Recipient email is required for export.")

    report_text = _build_report_text()
    subject = f"Clinical Insight Report - {datetime.now().strftime('%d %b %Y')}"

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    smtp_sender = os.getenv("SMTP_SENDER", smtp_user).strip()

    if smtp_host and smtp_user and smtp_pass and smtp_sender:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = smtp_sender
            msg["To"] = recipient_email
            msg.set_content(report_text)

            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            return render_home(export_success=f"Report sent to {recipient_email}.")
        except Exception as exc:
            return render_home(export_error=f"SMTP failed: {exc}")

    mailto_link = f"mailto:{quote(recipient_email)}?subject={quote(subject)}&body={quote(report_text)}"
    return redirect(mailto_link)


@app.route("/report/download", methods=["GET"])
@doctor_required
def download_report():
    report_text = _build_report_text()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    response = make_response(report_text)
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="clinical_report_{timestamp}.txt"'
    return response


if __name__ == "__main__":
    app.run(debug=True)
