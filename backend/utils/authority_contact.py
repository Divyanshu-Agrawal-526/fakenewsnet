import requests
import json
import logging
from typing import Dict, Any, List
import os
from datetime import datetime
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AuthorityContact:
    def __init__(self):
        self.api_keys = {
            'twilio': os.getenv('TWILIO_API_KEY', ''),
            'sendgrid': os.getenv('SENDGRID_API_KEY', '')
        }
        
        # Email configuration
        self.smtp_config = {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', '')
        }
        
        # Alert tracking
        self.alert_history = {}
        
        # Authority contact templates
        self.contact_templates = {
            'wildfire': {
                'subject': 'URGENT: Wildfire Alert',
                'priority': 'high',
                'template': 'wildfire_alert_template'
            },
            'flood': {
                'subject': 'URGENT: Flood Alert',
                'priority': 'high',
                'template': 'flood_alert_template'
            },
            'hurricane': {
                'subject': 'URGENT: Hurricane Alert',
                'priority': 'critical',
                'template': 'hurricane_alert_template'
            },
            'earthquake': {
                'subject': 'URGENT: Earthquake Alert',
                'priority': 'critical',
                'template': 'earthquake_alert_template'
            }
        }
    
    def send_alert(self, authority_id: str, message: str, location: str = None, disaster_type: str = None) -> Dict[str, Any]:
        """Send alert to specified authority"""
        try:
            alert_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Create alert record
            alert_record = {
                'alert_id': alert_id,
                'authority_id': authority_id,
                'message': message,
                'location': location,
                'disaster_type': disaster_type,
                'timestamp': timestamp,
                'status': 'pending'
            }
            
            # Get authority contact information
            authority_info = self.get_authority_info(authority_id)
            
            if not authority_info:
                return {
                    'success': False,
                    'message': 'Authority not found',
                    'alert_id': alert_id
                }
            
            # Prepare alert message
            alert_message = self.prepare_alert_message(message, location, disaster_type, authority_info)
            
            # Send alert through multiple channels
            results = {
                'email_sent': False,
                'sms_sent': False,
                'phone_sent': False
            }
            
            # Send email alert
            if authority_info.get('email'):
                results['email_sent'] = self.send_email_alert(
                    authority_info['email'],
                    alert_message,
                    disaster_type
                )
            
            # Send SMS alert
            if authority_info.get('phone') and authority_info['phone'] != '911':
                results['sms_sent'] = self.send_sms_alert(
                    authority_info['phone'],
                    alert_message
                )
            
            # Send phone alert (for critical disasters)
            if disaster_type in ['hurricane', 'earthquake'] and authority_info.get('phone'):
                results['phone_sent'] = self.send_phone_alert(
                    authority_info['phone'],
                    alert_message
                )
            
            # Update alert status
            if any(results.values()):
                alert_record['status'] = 'sent'
                success = True
                status_message = 'Alert sent successfully'
            else:
                alert_record['status'] = 'failed'
                success = False
                status_message = 'Failed to send alert'
            
            # Store alert record
            self.alert_history[alert_id] = alert_record
            
            return {
                'success': success,
                'message': status_message,
                'alert_id': alert_id,
                'results': results,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logging.error(f"Error sending alert: {e}")
            return {
                'success': False,
                'message': f'Error sending alert: {str(e)}',
                'alert_id': alert_id if 'alert_id' in locals() else None
            }
    
    def get_authority_info(self, authority_id: str) -> Dict[str, Any]:
        """Get authority contact information"""
        # In a real implementation, this would query a database
        # For now, return mock data
        authority_database = {
            'local_police': {
                'name': 'Local Police Department',
                'email': 'emergency@police.gov',
                'phone': '911',
                'type': 'Police'
            },
            'local_fire': {
                'name': 'Local Fire Department',
                'email': 'emergency@fire.gov',
                'phone': '911',
                'type': 'Fire'
            },
            'emergency_management': {
                'name': 'Emergency Management Agency',
                'email': 'emergency@ema.gov',
                'phone': '911',
                'type': 'Emergency Management'
            },
            'fire_department': {
                'name': 'Local Fire Department',
                'email': 'emergency@firedept.gov',
                'phone': '911',
                'type': 'Fire'
            },
            'forest_service': {
                'name': 'US Forest Service',
                'email': 'fire@fs.fed.us',
                'phone': '1-800-832-1355',
                'type': 'Forest Service'
            },
            'water_resources': {
                'name': 'Water Resources Department',
                'email': 'flood@water.gov',
                'phone': '1-800-555-0123',
                'type': 'Water Resources'
            },
            'national_weather_service': {
                'name': 'National Weather Service',
                'email': 'hurricane@weather.gov',
                'phone': '1-800-555-0124',
                'type': 'Weather Service'
            },
            'usgs': {
                'name': 'US Geological Survey',
                'email': 'earthquake@usgs.gov',
                'phone': '1-800-555-0125',
                'type': 'Geological Survey'
            }
        }
        
        return authority_database.get(authority_id, {})
    
    def prepare_alert_message(self, message: str, location: str, disaster_type: str, authority_info: Dict[str, Any]) -> str:
        """Prepare formatted alert message"""
        template = self.contact_templates.get(disaster_type, {})
        
        alert_message = f"""
URGENT ALERT - {template.get('subject', 'Emergency Alert')}

Authority: {authority_info.get('name', 'Unknown')}
Disaster Type: {disaster_type.title() if disaster_type else 'Unknown'}
Location: {location if location else 'Unknown'}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

MESSAGE:
{message}

This is an automated alert from the Fake News Detection System.
Please verify the information and take appropriate action.

For immediate assistance, contact 911 if this is a life-threatening emergency.

---
Alert ID: {str(uuid.uuid4())[:8]}
System: Fake News Detection During Natural Disasters
        """
        
        return alert_message.strip()
    
    def send_email_alert(self, email: str, message: str, disaster_type: str = None) -> bool:
        """Send email alert"""
        try:
            if not all([self.smtp_config['username'], self.smtp_config['password']]):
                logging.warning("SMTP credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = email
            msg['Subject'] = self.contact_templates.get(disaster_type, {}).get('subject', 'URGENT: Emergency Alert')
            
            # Add body
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email alert sent to {email}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending email alert: {e}")
            return False
    
    def send_sms_alert(self, phone: str, message: str) -> bool:
        """Send SMS alert using Twilio"""
        try:
            if not self.api_keys['twilio']:
                logging.warning("Twilio API key not configured")
                return False
            
            # In a real implementation, this would use Twilio API
            # For now, simulate SMS sending
            logging.info(f"SMS alert would be sent to {phone}: {message[:50]}...")
            return True
            
        except Exception as e:
            logging.error(f"Error sending SMS alert: {e}")
            return False
    
    def send_phone_alert(self, phone: str, message: str) -> bool:
        """Send phone alert for critical disasters"""
        try:
            # In a real implementation, this would use a phone API
            # For now, simulate phone alert
            logging.info(f"Phone alert would be sent to {phone}: {message[:50]}...")
            return True
            
        except Exception as e:
            logging.error(f"Error sending phone alert: {e}")
            return False
    
    def get_alert_status(self, alert_id: str) -> Dict[str, Any]:
        """Get status of a specific alert"""
        alert_record = self.alert_history.get(alert_id)
        
        if alert_record:
            return {
                'alert_id': alert_id,
                'status': alert_record['status'],
                'timestamp': alert_record['timestamp'],
                'authority_id': alert_record['authority_id'],
                'disaster_type': alert_record['disaster_type'],
                'location': alert_record['location']
            }
        else:
            return {
                'alert_id': alert_id,
                'status': 'not_found',
                'message': 'Alert not found'
            }
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        alerts = list(self.alert_history.values())
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts[:limit]
    
    def retry_failed_alert(self, alert_id: str) -> Dict[str, Any]:
        """Retry sending a failed alert"""
        alert_record = self.alert_history.get(alert_id)
        
        if not alert_record:
            return {
                'success': False,
                'message': 'Alert not found'
            }
        
        if alert_record['status'] != 'failed':
            return {
                'success': False,
                'message': 'Alert is not in failed status'
            }
        
        # Retry sending the alert
        return self.send_alert(
            alert_record['authority_id'],
            alert_record['message'],
            alert_record['location'],
            alert_record['disaster_type']
        )
    
    def cancel_alert(self, alert_id: str) -> Dict[str, Any]:
        """Cancel a pending alert"""
        alert_record = self.alert_history.get(alert_id)
        
        if not alert_record:
            return {
                'success': False,
                'message': 'Alert not found'
            }
        
        if alert_record['status'] != 'pending':
            return {
                'success': False,
                'message': 'Alert cannot be cancelled (not pending)'
            }
        
        alert_record['status'] = 'cancelled'
        
        return {
            'success': True,
            'message': 'Alert cancelled successfully',
            'alert_id': alert_id
        } 