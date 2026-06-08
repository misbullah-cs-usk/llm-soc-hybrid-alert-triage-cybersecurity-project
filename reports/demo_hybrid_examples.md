# Hybrid ML + Qwen Demo Examples

## Example 1

**True Label:** TruePositive

**ML Prediction:** TruePositive

**Correct:** True

**Alert Text:**

`Category: InitialAccess | MitreTechniques: T1566 | EntityType: Mailbox | EvidenceRole: Impacted | DetectorId: 789 | AlertTitle: 1079 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66`

**Analyst Summary:**

This alert indicates a potential initial access attempt through a mailbox, which could be an email phishing attack. The system has flagged this as a true positive.

**Risk Explanation:**

While the suspicion level is unknown, the alert involves a possible initial access method and should be investigated to determine if it represents a real threat or false positive. This can help in understanding the attacker's tactics and potentially prevent further malicious activities.

**Recommended Actions:**

["Review the mailbox activity logs for any unusual patterns or suspicious emails.", "Contact affected users to inform them of potential phishing attempts and provide guidance on how to identify such threats.", "Implement additional security measures, such as email filtering or training programs, to prevent future incidents.", "Monitor network traffic for any signs of lateral movement by the attacker."]

---

## Example 2

**True Label:** TruePositive

**ML Prediction:** TruePositive

**Correct:** True

**Alert Text:**

`Category: Exfiltration | MitreTechniques: unknown | EntityType: User | EvidenceRole: Impacted | DetectorId: 68 | AlertTitle: 453 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66`

**Analyst Summary:**

This alert indicates a potential exfiltration event involving user activity, with no specific MITRE ATT&CK technique identified.

**Risk Explanation:**

Exfiltration attempts can lead to sensitive data being stolen and potentially used for malicious purposes. Monitoring such activities is crucial as they may indicate an ongoing threat or compromise within the network.

**Recommended Actions:**

["Continue monitoring the user's activity to observe any further exfiltration attempts.", "Review logs for additional context on what data might have been exfiltrated and from which systems.", "Implement or enhance security controls such as network segmentation, encryption, and access controls to prevent future exfiltration events.", "Engage with incident response teams if the alert is confirmed to be a threat."]

---

## Example 3

**True Label:** FalsePositive

**ML Prediction:** FalsePositive

**Correct:** True

**Alert Text:**

`Category: Exfiltration | MitreTechniques: unknown | EntityType: File | EvidenceRole: Impacted | DetectorId: 357 | AlertTitle: 5519 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66`

**Analyst Summary:**

This alert is a false positive, indicating that the system incorrectly flagged this event as suspicious when it was not.

**Risk Explanation:**

Although this alert is labeled as a FalsePositive, it's important to monitor and understand why such events occur. This can help in improving the accuracy of future detections and ensure that real threats are not overlooked.

**Recommended Actions:**

["Review the detection rules for this specific event to identify any potential issues or misconfigurations.", "Monitor the affected system and network traffic for any unusual activity that might indicate a real threat.", "Consider adjusting the detection threshold settings if necessary, based on the frequency of false positives."]

---

## Example 4

**True Label:** BenignPositive

**ML Prediction:** BenignPositive

**Correct:** True

**Alert Text:**

`Category: Execution | MitreTechniques: T1559;T1106;T1059.005 | EntityType: File | EvidenceRole: Related | DetectorId: 39 | AlertTitle: 26 | SuspicionLevel: unknown | LastVerdict: unknown | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66`

**Analyst Summary:**

The alert is related to a file execution event involving multiple MITRE ATT&CK techniques, and the machine learning classifier has predicted it as benign. The analyst should investigate further to understand the context of this benign prediction.

**Risk Explanation:**

While the alert was classified as benign by the machine learning model, there are still potential risks associated with the observed file execution events involving multiple advanced attack techniques. It is important to verify if these activities align with normal operations or indicate malicious behavior.

**Recommended Actions:**

["Review the file's metadata and execution history to confirm its legitimacy.", "Check if any of the observed techniques align with known benign activities or legitimate software updates.", "Monitor for further suspicious activity related to this file.", "Consider retraining the machine learning model based on new data if necessary."]

---

## Example 5

**True Label:** TruePositive

**ML Prediction:** BenignPositive

**Correct:** False

**Alert Text:**

`Category: SuspiciousActivity | MitreTechniques: unknown | EntityType: Process | EvidenceRole: Related | DetectorId: 3 | AlertTitle: 4 | SuspicionLevel: Suspicious | LastVerdict: Suspicious | CountryCode: 242 | State: 1445 | City: 10630 | OSFamily: 5 | OSVersion: 66`

**Analyst Summary:**

A suspicious process was detected, which is related to a previous alert with the same title and suspicion level. The process appears to be benign based on the machine learning classifier.

**Risk Explanation:**

Although this alert has been classified as benign, it's still important for an analyst to review the details of the process to ensure there are no hidden malicious activities or anomalies that could indicate a potential threat in the future.

**Recommended Actions:**

["Review the process and its behavior for any unusual activity", "Monitor the process over time to see if it exhibits any malicious patterns", "Check if there are any known vulnerabilities associated with this process or OS version", "Consider isolating the system temporarily until further investigation is completed"]

---

