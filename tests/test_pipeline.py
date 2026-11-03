"""
End-to-End Test Suite for FIREGUARD AI
Verifies model loading, inference on benchmark images, history management, and CSV export.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from models.model_loader import load_fire_model
from models.predict import run_vgg16_inference
from utils.history_manager import history_manager

def test_pipeline():
    print("=" * 60)
    print("RUNNING FIREGUARD AI VERIFICATION TEST SUITE")
    print("=" * 60)

    # 1. Model Loading
    status = load_fire_model()
    assert status.is_loaded, f"Model failed to load: {status.status_message}"
    print(f"[PASS] Model Loaded Successfully: {status.weights_path}")
    print(f"[PASS] Supported Classes: {status.classes}")

    # 2. Test Inferences
    sample_files = [
        "cctv_fire_scene.jpg",
        "cctv_smoke_scene.jpg",
        "cctv_normal_room.jpg"
    ]
    for sname in sample_files:
        path = config.SAMPLE_DIR / sname
        assert path.exists(), f"Sample image {path} does not exist!"
        result = run_vgg16_inference(str(path), status)
        assert result["success"], f"Inference failed on {sname}: {result.get('message')}"
        print(f"[PASS] {sname:22} -> Predicted: {result['predicted_class']:8} | Confidence: {result['confidence_percentage']:5.1f}% | Latency: {result['latency_ms']}ms")

        # 3. Add to Audit Trail
        history_manager.add_record(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            filename=sname,
            camera_id="Verification Test Suite",
            notes="Automated Pipeline Test"
        )

    # 4. History and Stats Check
    stats = history_manager.get_statistics()
    assert stats["total"] >= 3, f"Expected at least 3 records, got {stats['total']}"
    print(f"[PASS] Audit Trail Statistics: Total={stats['total']}, Fire={stats['fire_count']}, Normal={stats['normal_count']}, AvgConf={stats['avg_confidence']}%")

    # 5. CSV Export Check
    csv_data = history_manager.export_to_csv()
    assert len(csv_data.splitlines()) >= 4, "CSV export should contain header and records"
    print(f"[PASS] CSV Export Generated ({len(csv_data)} bytes)")

    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
