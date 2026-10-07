def analyze_severity(label, confidence):
    """
    Analyze the severity of a brain tumor prediction.

    Args:
        label (str): Predicted tumor type (e.g., 'Glioma', 'Meningioma', 'Pituitary', 'No Tumor').
        confidence (float): Prediction confidence as a percentage (0-100).

    Returns:
        dict: Severity analysis with 'severity', 'risk_level', 'description', and 'recommendations'.
    """
    label_lower = label.lower().strip()

    # No tumor detected
    if label_lower in ("no tumor", "notumor", "no_tumor"):
        return {
            "severity": "None",
            "risk_level": "Low",
            "description": "No tumor detected in the MRI scan.",
            "recommendations": [
                "Continue regular health checkups.",
                "Maintain a healthy lifestyle.",
                "Consult a neurologist if symptoms persist."
            ],
            "color": "#28a745"
        }

    # Tumor severity based on type and confidence
    severity_map = {
        "glioma": {
            "high_conf": {
                "severity": "Critical",
                "risk_level": "Very High",
                "description": (
                    "Glioma detected with high confidence. Gliomas are tumors that arise from glial cells "
                    "in the brain. They can be aggressive (high-grade) or slow-growing (low-grade). "
                    "Immediate medical attention is strongly recommended."
                ),
                "recommendations": [
                    "Seek immediate consultation with a neuro-oncologist.",
                    "Additional imaging (contrast MRI, MR spectroscopy) is recommended.",
                    "Biopsy may be required to determine the grade.",
                    "Discuss treatment options including surgery, radiation, and chemotherapy.",
                    "Consider getting a second opinion from a specialized center."
                ],
                "color": "#dc3545"
            },
            "low_conf": {
                "severity": "High",
                "risk_level": "High",
                "description": (
                    "Possible glioma detected but with lower confidence. Further diagnostic tests "
                    "are recommended to confirm the finding."
                ),
                "recommendations": [
                    "Schedule a follow-up MRI with contrast enhancement.",
                    "Consult with a neurologist for further evaluation.",
                    "Monitor for symptoms such as headaches, seizures, or cognitive changes.",
                    "Consider advanced imaging techniques for confirmation."
                ],
                "color": "#e74c3c"
            }
        },
        "meningioma": {
            "high_conf": {
                "severity": "Moderate",
                "risk_level": "Moderate",
                "description": (
                    "Meningioma detected with high confidence. Meningiomas are typically benign tumors "
                    "that arise from the meninges (the membranes surrounding the brain). Most meningiomas "
                    "are slow-growing and may not require immediate treatment."
                ),
                "recommendations": [
                    "Consult with a neurosurgeon for evaluation.",
                    "Regular monitoring with periodic MRI scans.",
                    "Treatment may include observation, surgery, or radiation therapy.",
                    "Watch for symptoms like headaches, vision changes, or weakness.",
                    "Most meningiomas have a good prognosis with proper management."
                ],
                "color": "#fd7e14"
            },
            "low_conf": {
                "severity": "Moderate",
                "risk_level": "Moderate",
                "description": (
                    "Possible meningioma detected but with lower confidence. Additional imaging "
                    "is recommended to confirm the diagnosis."
                ),
                "recommendations": [
                    "Schedule a contrast-enhanced MRI for better characterization.",
                    "Consult with a neurologist.",
                    "Monitor any neurological symptoms.",
                    "Follow up within 3-6 months."
                ],
                "color": "#f39c12"
            }
        },
        "pituitary": {
            "high_conf": {
                "severity": "Moderate",
                "risk_level": "Moderate",
                "description": (
                    "Pituitary tumor detected with high confidence. Pituitary tumors (adenomas) are "
                    "usually benign growths on the pituitary gland. They can affect hormone production "
                    "and may cause vision problems if they grow large enough."
                ),
                "recommendations": [
                    "Consult with an endocrinologist and neurosurgeon.",
                    "Hormonal blood tests are recommended.",
                    "Visual field testing may be needed.",
                    "Treatment options include medication, surgery, or radiation.",
                    "Many pituitary tumors respond well to medication alone."
                ],
                "color": "#fd7e14"
            },
            "low_conf": {
                "severity": "Low",
                "risk_level": "Low-Moderate",
                "description": (
                    "Possible pituitary tumor detected with lower confidence. Further endocrine "
                    "evaluation and imaging are recommended."
                ),
                "recommendations": [
                    "Schedule hormonal panel blood tests.",
                    "Follow-up MRI with pituitary protocol.",
                    "Consult with an endocrinologist.",
                    "Monitor for hormonal symptoms (fatigue, weight changes, vision issues)."
                ],
                "color": "#f39c12"
            }
        }
    }

    # Determine high vs low confidence threshold
    conf_key = "high_conf" if confidence >= 70 else "low_conf"

    # Look up severity info
    if label_lower in severity_map:
        return severity_map[label_lower][conf_key]

    # Fallback for unknown labels
    return {
        "severity": "Unknown",
        "risk_level": "Unknown",
        "description": f"Detected: {label} with {confidence:.1f}% confidence. Please consult a medical professional.",
        "recommendations": [
            "Consult with a neurologist immediately.",
            "Bring this report to your healthcare provider.",
            "Additional diagnostic imaging may be needed."
        ],
        "color": "#6c757d"
    }
