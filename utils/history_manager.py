"""
Detection History and Session Storage Manager for FIREGUARD AI
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import config

class HistoryManager:
    """Manages persistent and session-based detection audit logs."""

    def __init__(self, storage_path: Path = config.HISTORY_FILE):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Creates the JSON storage file if missing."""
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def load_records(self) -> List[Dict[str, Any]]:
        """Loads all detection records from persistent JSON file."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                return sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception:
            return []

    def save_records(self, records: List[Dict[str, Any]]) -> None:
        """Saves all records back to JSON file."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def add_record(
        self,
        predicted_class: str,
        confidence: float,
        filename: str,
        camera_id: str = "Manual Upload",
        notes: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adds a genuine prediction result to detection history.
        Does not fabricate detections.
        """
        records = self.load_records()
        now = datetime.now()
        
        # Determine alert status
        class_lower = predicted_class.lower()
        if "fire" in class_lower:
            alert_status = "EMERGENCY"
        elif "smoke" in class_lower:
            alert_status = "WARNING"
        else:
            alert_status = "NORMAL"

        record_id = f"DET-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        record = {
            "id": record_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4),
            "filename": filename,
            "camera_id": camera_id,
            "alert_status": alert_status,
            "notes": notes,
            "details": details or {}
        }
        
        records.insert(0, record)
        self.save_records(records)
        return record

    def delete_record(self, record_id: str) -> bool:
        """Deletes a record by its unique ID."""
        records = self.load_records()
        initial_len = len(records)
        updated = [r for r in records if r.get("id") != record_id]
        if len(updated) != initial_len:
            self.save_records(updated)
            return True
        return False

    def clear_history(self) -> None:
        """Clears all records."""
        self.save_records([])

    def export_to_csv(self) -> str:
        """Exports records as a CSV string."""
        records = self.load_records()
        if not records:
            return ""
        # Flatten details for clean tabular export
        rows = []
        for r in records:
            rows.append({
                "Detection ID": r.get("id"),
                "Timestamp": r.get("timestamp"),
                "Predicted Class": r.get("predicted_class"),
                "Confidence (%)": f"{r.get('confidence', 0)*100:.2f}%",
                "Raw Confidence": r.get("confidence"),
                "Filename": r.get("filename"),
                "Source / Camera": r.get("camera_id"),
                "Alert Status": r.get("alert_status"),
                "Notes": r.get("notes")
            })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates live metrics from actual detection records."""
        records = self.load_records()
        total = len(records)
        if total == 0:
            return {
                "total": 0,
                "fire_count": 0,
                "smoke_count": 0,
                "normal_count": 0,
                "avg_confidence": 0.0,
                "emergency_rate": 0.0
            }

        fire_count = sum(1 for r in records if "fire" in r.get("predicted_class", "").lower())
        smoke_count = sum(1 for r in records if "smoke" in r.get("predicted_class", "").lower())
        normal_count = sum(1 for r in records if "normal" in r.get("predicted_class", "").lower() or "non-fire" in r.get("predicted_class", "").lower())
        
        confidences = [r.get("confidence", 0) for r in records if r.get("confidence") is not None]
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        return {
            "total": total,
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "normal_count": normal_count,
            "avg_confidence": round(avg_conf * 100, 2),
            "emergency_rate": round((fire_count / total) * 100, 2)
        }

# Global singleton instance
history_manager = HistoryManager()
