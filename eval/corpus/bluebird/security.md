# Security and Compliance

## Encryption

All data is encrypted at rest using AES-256. Data in transit uses TLS 1.2 or higher.
Encryption keys are rotated every 90 days.

## Access Control

Multi-factor authentication is required for all admin accounts.
User sessions expire after 8 hours of inactivity.

## Audit Logs

Audit logs are retained for 12 months. Logs older than 12 months are archived
to cold storage for an additional 24 months before permanent deletion.

## Anonymity Floor

Individual responses are never surfaced to managers. Aggregated results require
a minimum of 8 respondents before display. Groups smaller than this are suppressed.

## Penetration Testing

Bluebird undergoes external penetration testing twice per year.
Results are available to Enterprise customers under NDA within 30 days of completion.

## Incident Response

In the event of a security incident, affected customers will be notified within
72 hours of confirmed detection, in compliance with GDPR Article 33.
