from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import traceback
import io
import csv
from datetime import datetime

from app.models import db  # ✅ Shared DB instance
from app.models.schema import PredictionBatch, Prediction, SinglePrediction  # ✅ Correct model definitions
from app.services.predictor import predict_label, predict_labels

dashboard_bp = Blueprint("dashboard", __name__)

# Mock data for visualization
mock_data = {
    "labels": ["Anxiety", "Bipolar", "Depression", "Normal", "Personality disorder", "Stress", "Suicidal"],
    "counts": [768, 556, 3081, 3269, 215, 517, 2131]
}


@dashboard_bp.route("/", methods=["GET", "POST"])
def dashboard():
    prediction_result = None
    batch_results = None
    batch_id = None

    if request.method == "POST":
        try:
            # Handle single prediction
            if "text" in request.form and request.form["text"].strip():
                user_input = request.form["text"]
                prediction_result = predict_label(user_input)

                single_pred = SinglePrediction(
                    text=user_input,
                    predicted_label=prediction_result
                )
                db.session.add(single_pred)
                db.session.commit()

                flash(f"Prediction: {prediction_result}", "success")

            # Handle file upload
            elif "file" in request.files:
                file = request.files["file"]
                print(f"[DEBUG] File received: {file.filename}")

                if file.filename == "":
                    flash("No file selected.", "error")
                elif not file.filename.lower().endswith('.csv'):
                    flash("Please upload a CSV file.", "error")
                else:
                    df = None
                    encodings_tried = []
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, encoding=encoding)
                            print(f"[DEBUG] DataFrame loaded with encoding {encoding}, shape: {df.shape}")
                            break
                        except Exception as e:
                            encodings_tried.append(f"{encoding}: {e}")
                            print(f"[DEBUG] Failed to load with encoding {encoding}: {e}")
                            continue

                    if df is None:
                        flash("Could not read the CSV file. Tried encodings: " + "; ".join(encodings_tried), "error")
                        return render_template("base.html")

                    if "text" not in df.columns:
                        print(f"[DEBUG] 'text' column not found in columns: {df.columns}")
                        flash("CSV must have a 'text' column.", "error")
                        return render_template("base.html")

                    texts = df["text"].dropna().astype(str).tolist()
                    print(f"[DEBUG] Extracted {len(texts)} texts from CSV.")

                    if not texts:
                        flash("No valid text found in file.", "error")
                    else:
                        if len(texts) > 1000:
                            texts = texts[:1000]
                            flash("Processing only first 1000 rows.", "info")
                        print(f"[DEBUG] Sending {len(texts)} texts to predict_labels.")
                        predictions = predict_labels(texts)
                        print(f"[DEBUG] Received {len(predictions)} predictions.")
                        batch_results = list(zip(texts, predictions))

                        batch = PredictionBatch(
                            filename=file.filename,
                            total_predictions=len(batch_results)
                        )
                        db.session.add(batch)
                        db.session.flush()

                        for text, prediction in batch_results:
                            pred = Prediction(
                                batch_id=batch.id,
                                original_text=text,
                                predicted_label=prediction
                            )
                            db.session.add(pred)

                        db.session.commit()
                        batch_id = batch.id

                        flash(f"Processed {len(batch_results)} texts.", "success")
                        flash(f"Batch ID: {batch_id}", "info")

        except Exception as e:
            print(traceback.format_exc())
            flash(f"An error occurred: {str(e)}", "error")

    recent_batches = PredictionBatch.query.order_by(PredictionBatch.created_at.desc()).limit(10).all()

    return render_template(
        "base.html",
        prediction=prediction_result,
        batch_results=batch_results,
        batch_id=batch_id,
        recent_batches=recent_batches,
        labels=mock_data["labels"],
        counts=mock_data["counts"],
        label_count_pairs=zip(mock_data["labels"], mock_data["counts"])
    )


@dashboard_bp.route("/download/<int:batch_id>")
def download_batch(batch_id):
    try:
        batch = PredictionBatch.query.get_or_404(batch_id)
        predictions = Prediction.query.filter_by(batch_id=batch_id).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Text', 'Predicted_Label', 'Prediction_Time'])

        for pred in predictions:
            writer.writerow([
                pred.original_text,
                pred.predicted_label,
                pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])

        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)

        filename = f"predictions_{batch.filename}_{batch.created_at.strftime('%Y%m%d_%H%M%S')}.csv"

        return send_file(mem, as_attachment=True, download_name=filename, mimetype='text/csv')

    except Exception as e:
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(url_for('dashboard.dashboard'))


@dashboard_bp.route("/download_all_single")
def download_all_single():
    try:
        predictions = SinglePrediction.query.order_by(SinglePrediction.created_at.desc()).all()

        if not predictions:
            flash("No single predictions found.", "error")
            return redirect(url_for('dashboard.dashboard'))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Text', 'Predicted_Label', 'Prediction_Time'])

        for pred in predictions:
            writer.writerow([
                pred.text,
                pred.predicted_label,
                pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])

        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)

        filename = f"single_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return send_file(mem, as_attachment=True, download_name=filename, mimetype='text/csv')

    except Exception as e:
        flash(f"Error downloading file: {str(e)}", "error")
        return redirect(url_for('dashboard.dashboard'))


@dashboard_bp.route("/history")
def prediction_history():
    batches = PredictionBatch.query.order_by(PredictionBatch.created_at.desc()).all()
    single_predictions = SinglePrediction.query.order_by(SinglePrediction.created_at.desc()).limit(50).all()

    return render_template(
        "history.html",
        batches=batches,
        single_predictions=single_predictions
    )
