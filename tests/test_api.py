import os
import time
import unittest
from pathlib import Path

# Placeholder frames keep the API tests fast and independent of GPUs and API keys.
os.environ["STORYBOARD_BACKEND"] = "fallback"

from fastapi.testclient import TestClient

from src.api.main import app

SAMPLE = (Path(__file__).parent.parent / "examples" / "sample_script.txt").read_bytes()


def upload(data: bytes):
    return {"file": ("script.txt", data, "text/plain")}


class ApiTests(unittest.TestCase):
    client = TestClient(app)

    def test_upload_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('<input id="file"', response.text)

    def test_analyze_returns_scenes_with_shots(self):
        response = self.client.post("/analyze-script", files=upload(SAMPLE))
        self.assertEqual(response.status_code, 200)
        scenes = response.json()["scenes"]
        self.assertEqual([s["id"] for s in scenes], [1, 2, 3])
        self.assertEqual(len(scenes[0]["shots"]), 2)

    def test_bad_uploads_are_400(self):
        self.assertEqual(self.client.post("/analyze-script", files=upload(b"\xff\xfe\x00bad")).status_code, 400)
        self.assertEqual(self.client.post("/jobs", files=upload(b"   \n")).status_code, 400)

    def test_job_reports_progress_and_serves_pdf(self):
        job = self.client.post("/jobs", files=upload(SAMPLE)).json()
        self.assertEqual(job["total"], 7)  # 2 + 2 + 3 shots in the sample script

        deadline = time.time() + 60
        while (status := self.client.get(f"/jobs/{job['id']}").json())["status"] not in ("done", "failed"):
            self.assertLess(time.time(), deadline, "job did not finish")
            time.sleep(0.2)
        self.assertEqual(status["status"], "done", status["error"])
        self.assertEqual(status["done"], 7)

        pdf = self.client.get(f"/jobs/{job['id']}/pdf")
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF"))

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/jobs/nope").status_code, 404)
        self.assertEqual(self.client.get("/jobs/nope/pdf").status_code, 404)

    def test_generate_storyboard_returns_pdf(self):
        response = self.client.post("/generate-storyboard", files=upload(SAMPLE))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
