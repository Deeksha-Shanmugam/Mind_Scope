from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import pandas as pd
from app.services.predictor import predict_label, predict_labels
import traceback

dashboard_bp = Blueprint("dashboard", __name__)

# Sample placeholder for visualization data
mock_data = {
    "labels": ["Anxiety", "Bipolar", "Depression", "Normal", "Personality disorder", "Stress", "Suicidal"],
    "counts": [768, 556, 3081, 3269, 215, 517, 2131]
}

@dashboard_bp.route("/", methods=["GET", "POST"])
def dashboard():
    prediction_result = None
    batch_results = None
    
    if request.method == "POST":
        print("POST request received")  # Debug log
        print("Form data:", request.form)  # Debug log
        print("Files:", request.files)  # Debug log
        
        try:
            # Handle single text prediction
            if "text" in request.form and request.form["text"].strip():
                user_input = request.form["text"]
                print(f"Processing text: {user_input[:50]}...")  # Debug log
                prediction_result = predict_label(user_input)
                print(f"Prediction result: {prediction_result}")  # Debug log
                flash(f"Prediction: {prediction_result}", "success")
                
            # Handle file upload
            elif "file" in request.files:
                file = request.files["file"]
                print(f"File received: {file.filename}")  # Debug log
                
                if file.filename == "":
                    flash("No file selected.", "error")
                elif not file.filename.lower().endswith('.csv'):
                    flash("Please upload a CSV file.", "error")
                else:
                    try:
                        # Try to read CSV with different encodings
                        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                        df = None
                        
                        for encoding in encodings_to_try:
                            try:
                                file.seek(0)  # Reset file pointer
                                df = pd.read_csv(file, encoding=encoding)
                                print(f"Successfully read CSV with {encoding} encoding")
                                break
                            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                                print(f"Failed to read with {encoding} encoding: {str(e)}")
                                continue
                            except Exception as e:
                                print(f"Unexpected error with {encoding} encoding: {str(e)}")
                                continue
                        
                        # If all encodings failed, try with error handling
                        if df is None:
                            try:
                                file.seek(0)
                                df = pd.read_csv(file, encoding='utf-8', errors='ignore')
                                print("Successfully read CSV with UTF-8 and error ignore")
                            except Exception as e:
                                print(f"Final attempt failed: {str(e)}")
                                flash("Could not read the CSV file. Please check the file format and encoding.", "error")
                                return render_template(
                                    "base.html",
                                    prediction=prediction_result,
                                    batch_results=batch_results,
                                    labels=mock_data["labels"],
                                    counts=mock_data["counts"],
                                    label_count_pairs=zip(mock_data["labels"], mock_data["counts"])
                                )
                        
                        print(f"CSV shape: {df.shape}")  # Debug log
                        print(f"CSV columns: {list(df.columns)}")  # Debug log
                        
                        # Show first few rows for debugging
                        if len(df) > 0:
                            print(f"First row: {df.iloc[0].to_dict()}")
                        else:
                            print("CSV is empty")
                        
                        if "text" not in df.columns:
                            flash("CSV must have a 'text' column.", "error")
                            available_cols = ", ".join(df.columns)
                            flash(f"Available columns: {available_cols}", "info")
                        else:
                            # Process the texts
                            texts = df["text"].dropna().astype(str).tolist()
                            print(f"Processing {len(texts)} texts...")  # Debug log
                            
                            if len(texts) == 0:
                                flash("No valid text data found in the CSV file.", "error")
                            else:
                                # Limit to first 100 rows for performance
                                if len(texts) > 100:
                                    texts = texts[:100]
                                    flash(f"Processing first 100 rows only.", "info")
                                
                                predictions = predict_labels(texts)
                                batch_results = list(zip(texts, predictions))
                                print(f"Batch processing complete: {len(batch_results)} results")  # Debug log
                                flash(f"Successfully processed {len(batch_results)} texts.", "success")
                                
                    except pd.errors.EmptyDataError:
                        flash("The uploaded file is empty.", "error")
                    except pd.errors.ParserError as e:
                        flash(f"Error parsing CSV file: {str(e)}", "error")
                    except Exception as e:
                        print(f"File processing error: {str(e)}")  # Debug log
                        print(traceback.format_exc())  # Debug log
                        flash(f"Error processing file: {str(e)}", "error")
            else:
                flash("Please enter text or upload a CSV file.", "error")
                
        except Exception as e:
            print(f"General error: {str(e)}")  # Debug log
            print(traceback.format_exc())  # Debug log
            flash(f"An error occurred: {str(e)}", "error")
    
    return render_template(
        "base.html",
        prediction=prediction_result,
        batch_results=batch_results,
        labels=mock_data["labels"],
        counts=mock_data["counts"],
        label_count_pairs=zip(mock_data["labels"], mock_data["counts"])
    )