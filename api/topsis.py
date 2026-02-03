from http.server import BaseHTTPRequestHandler
import json
import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import cgi
import pandas as pd
import numpy as np


def run_topsis(csv_content, weights_str, impacts_str):
    """
    Run TOPSIS algorithm on the provided CSV content.
    Returns the result as CSV string.
    """
    # Read CSV from string
    data = pd.read_csv(io.StringIO(csv_content))
    
    if data.shape[1] < 3:
        raise Exception("Input file must contain at least 3 columns")
    
    criteria_data = data.iloc[:, 1:]
    
    # Parse weights and impacts
    weights = np.array(weights_str.split(","), dtype=float)
    impacts = [i.strip() for i in impacts_str.split(",")]
    
    if len(weights) != criteria_data.shape[1]:
        raise Exception(f"Weights count ({len(weights)}) must match criteria columns ({criteria_data.shape[1]})")
    
    if len(impacts) != criteria_data.shape[1]:
        raise Exception(f"Impacts count ({len(impacts)}) must match criteria columns ({criteria_data.shape[1]})")
    
    for imp in impacts:
        if imp not in ["+", "-"]:
            raise Exception("Impacts must be + or -")
    
    # Normalize matrix
    norm_matrix = criteria_data / np.sqrt((criteria_data ** 2).sum())
    
    # Weighted matrix
    weighted_matrix = norm_matrix * weights
    
    # Ideal best and worst
    ideal_best, ideal_worst = [], []
    
    for i in range(criteria_data.shape[1]):
        if impacts[i] == "+":
            ideal_best.append(weighted_matrix.iloc[:, i].max())
            ideal_worst.append(weighted_matrix.iloc[:, i].min())
        else:
            ideal_best.append(weighted_matrix.iloc[:, i].min())
            ideal_worst.append(weighted_matrix.iloc[:, i].max())
    
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    
    # Calculate distances
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate score and rank
    score = dist_worst / (dist_best + dist_worst)
    rank = score.rank(ascending=False)
    
    data["Topsis Score"] = score
    data["Rank"] = rank.astype(int)
    
    # Return as CSV string
    return data.to_csv(index=False)


def send_email(to_email, result_csv, original_filename):
    """
    Send the result CSV via email.
    """
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_pass = os.environ.get('SMTP_PASS', '')
    
    if not smtp_user or not smtp_pass:
        raise Exception("Email configuration not set. Please configure SMTP_USER and SMTP_PASS environment variables.")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = 'TOPSIS Analysis Results'
    
    # Email body
    body = """
Hello,

Your TOPSIS analysis has been completed successfully.

Please find the results attached as a CSV file. The file includes:
- Original data columns
- Topsis Score (higher is better)
- Rank (1 = best alternative)

Thank you for using our TOPSIS Web Service!

Best regards,
TOPSIS Web Service
"""
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach CSV file
    result_filename = original_filename.replace('.csv', '_result.csv') if original_filename else 'topsis_result.csv'
    attachment = MIMEBase('text', 'csv')
    attachment.set_payload(result_csv)
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename="{result_filename}"')
    msg.attach(attachment)
    
    # Send email
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def parse_multipart(handler):
    """Parse multipart form data from the request."""
    content_type = handler.headers.get('Content-Type', '')
    
    if 'multipart/form-data' not in content_type:
        raise Exception("Content-Type must be multipart/form-data")
    
    # Get boundary
    boundary = None
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary = part[9:]
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            break
    
    if not boundary:
        raise Exception("No boundary found in Content-Type")
    
    # Read body
    content_length = int(handler.headers.get('Content-Length', 0))
    body = handler.rfile.read(content_length)
    
    # Parse fields
    fields = {}
    file_content = None
    file_name = None
    
    boundary_bytes = ('--' + boundary).encode()
    parts = body.split(boundary_bytes)
    
    for part in parts:
        if not part or part == b'--\r\n' or part == b'--':
            continue
        
        # Split header and content
        if b'\r\n\r\n' in part:
            header, content = part.split(b'\r\n\r\n', 1)
        elif b'\n\n' in part:
            header, content = part.split(b'\n\n', 1)
        else:
            continue
        
        header = header.decode('utf-8', errors='ignore')
        
        # Get field name
        name = None
        filename = None
        for line in header.split('\n'):
            line = line.strip()
            if 'Content-Disposition:' in line or 'content-disposition:' in line.lower():
                parts_cd = line.split(';')
                for p in parts_cd:
                    p = p.strip()
                    if p.startswith('name='):
                        name = p[5:].strip('"\'')
                    elif p.startswith('filename='):
                        filename = p[9:].strip('"\'')
        
        # Clean content (remove trailing CRLF)
        content = content.rstrip(b'\r\n')
        
        if filename:
            file_content = content.decode('utf-8', errors='ignore')
            file_name = filename
        elif name:
            fields[name] = content.decode('utf-8', errors='ignore')
    
    return fields, file_content, file_name


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse multipart form data
            fields, file_content, file_name = parse_multipart(self)
            
            weights = fields.get('weights', '')
            impacts = fields.get('impacts', '')
            email = fields.get('email', '')
            
            # Validate inputs
            if not file_content:
                raise Exception("No file uploaded")
            if not weights:
                raise Exception("Weights are required")
            if not impacts:
                raise Exception("Impacts are required")
            if not email:
                raise Exception("Email is required")
            
            # Run TOPSIS
            result_csv = run_topsis(file_content, weights, impacts)
            
            # Send email
            send_email(email, result_csv, file_name)
            
            # Return success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'success': True,
                'message': f'TOPSIS analysis completed! Results sent to {email}'
            }
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(response).encode())
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'message': 'TOPSIS API is running. Use POST to submit analysis.'
        }
        self.wfile.write(json.dumps(response).encode())
