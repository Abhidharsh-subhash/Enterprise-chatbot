from app.core.celery_app import celery_app
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings


@celery_app.task
def send_email_task(to_email: str, subject: str, html_content: str):
    """Background task for sending email."""
    try:
        msg = MIMEMultipart()
        msg["From"] = settings.sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))

        smtp = smtplib.SMTP(settings.smtp_server, settings.smtp_port)
        smtp.starttls()
        smtp.login(settings.sender_email, settings.sender_password)
        smtp.send_message(msg)
        smtp.quit()

        return {"status": "success", "to": to_email}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
