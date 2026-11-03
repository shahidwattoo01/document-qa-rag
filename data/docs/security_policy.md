# Lumora Systems Information Security Policy

*Fictional sample document used for the Document Q&A with RAG demo.*

## 1. Scope

This policy applies to all employees, contractors, and third parties who access Lumora Systems information, systems, or networks. Violations may lead to disciplinary action, up to and including termination of employment or contract.

## 2. Data Classification

Lumora classifies information into four levels:

- **Public** — information approved for public release, such as marketing material.
- **Internal** — day-to-day business information that is not meant for the public, such as internal wikis.
- **Confidential** — sensitive business data such as financial forecasts, source code, and customer contracts.
- **Restricted** — the most sensitive data, including customer personal data, payment data, and encryption keys.

Restricted data must be encrypted at rest using AES-256 and in transit using TLS 1.2 or higher. Restricted data may never be stored on personal devices or in personal cloud accounts.

## 3. Passwords and Authentication

Passwords for company accounts must be at least 14 characters long. Passwords do not need to be rotated on a fixed schedule, but must be changed immediately if a compromise is suspected. All employees must store passwords in the company-approved password manager.

Multi-factor authentication (MFA) is mandatory for all company systems. Hardware security keys are required for administrators and for anyone with access to production infrastructure. SMS-based codes are not an accepted MFA method.

## 4. Device Security

Company laptops must have full-disk encryption enabled and must lock automatically after 5 minutes of inactivity. Operating system security updates must be installed within 14 days of release; critical updates must be installed within 72 hours. The endpoint protection agent must not be disabled.

## 5. Access Control

Access is granted on the principle of least privilege. Managers review their team's access rights every quarter. When an employee leaves the company, all of their access must be revoked within 4 hours of their last working moment. Shared accounts are prohibited except for documented service accounts.

## 6. Incident Response

A security incident is any event that could compromise the confidentiality, integrity, or availability of Lumora data or systems. Suspected incidents must be reported to the Security team within 1 hour of discovery through the #security-incidents channel or by emailing the security hotline.

Incidents are classified by severity:

- **SEV-1** — active breach of Restricted data or full production outage. The incident commander must be paged immediately and executive leadership is informed within 2 hours.
- **SEV-2** — confirmed compromise of a single system or account without evidence of data exfiltration.
- **SEV-3** — suspicious activity or policy violation with no confirmed impact.

If a breach involves customer personal data, the Legal team must be notified so that regulators can be informed within 72 hours where required by law. A written post-incident review must be completed within 10 business days of resolving any SEV-1 or SEV-2 incident.

## 7. Phishing and Social Engineering

Employees must report suspected phishing emails using the "Report Phish" button in the mail client. Lumora runs simulated phishing campaigns every quarter. Employees who fail two simulations within twelve months must complete additional security awareness training.

## 8. Security Training

All employees must complete security awareness training within their first 30 days and then once every year.
