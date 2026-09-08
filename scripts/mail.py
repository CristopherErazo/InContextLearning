import smtplib
from email.message import EmailMessage

SMTP_USER = "cris.erv.98@gmail.com"
SMTP_PASS = "password"
TO = "cerazova@sissa.it"

msg = EmailMessage()
msg["Subject"] = "Test email from Python"
msg["From"] = SMTP_USER
msg["To"] = TO
msg.set_content("""
Hello from the monitor script.
This is a test email.
I hope you are doing well.
:) 
""")

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(SMTP_USER, SMTP_PASS)
    server.send_message(msg)

print("Email sent")