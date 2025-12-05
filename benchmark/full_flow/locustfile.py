import time
from locust import HttpUser, task


class FullFlowUser(HttpUser):
    """
    Mimics the full frontend flow:
    1. Upload MP3 to /api/upload-and-search
    2. Poll /api/results/{upload_id} until completed or failed

    Reports the entire flow as a single "Full Workflow" metric.
    """

    # Maximum time to poll for results (in seconds)
    MAX_POLL_TIME = 60
    # Interval between poll requests (in seconds)
    POLL_INTERVAL = 1

    @task
    def full_workflow(self):
        start_time = time.time()

        # Step 1: Upload MP3 file & trigger processing
        upload_id = self._upload_file()

        if upload_id:
            # Step 2: Poll for results
            success, error_message = self._poll_results(upload_id)
        else:
            success = False
            error_message = "Upload failed"

        # Report the full workflow as a single grouped metric
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        self.environment.events.request.fire(
            request_type="WORKFLOW",
            name="Full Workflow (Upload + Poll)",
            response_time=total_time,
            response_length=0,
            exception=Exception(error_message) if not success else None,
            context={},
        )

    def _upload_file(self) -> str | None:
        """Upload test_track.mp3 to /api/upload-and-search"""
        try:
            with open("test_track.mp3", "rb") as mp3_file:
                files = {"file": ("test_track.mp3", mp3_file, "audio/mpeg")}

                with self.client.post(
                    "/api/upload-and-search",
                    files=files,
                    name="/api/upload-and-search",
                    catch_response=True,
                ) as response:
                    if response.status_code == 200:
                        data = response.json()
                        upload_id = data.get("upload_id")

                        if upload_id:
                            response.success()
                            return upload_id
                        else:
                            response.failure("No upload_id in response")
                            return None
                    else:
                        response.failure(f"Upload failed: {response.status_code}")
                        return None

        except FileNotFoundError:
            print("Error: test_track.mp3 not found in benchmark/full_flow/")
            return None
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def _poll_results(self, upload_id: str) -> tuple[bool, str | None]:
        """Poll /api/results/{upload_id} until completed, failed, or timeout

        Returns:
            tuple: (success: bool, error_message: str | None)
        """
        start_time = time.time()
        attempts = 0

        while time.time() - start_time < self.MAX_POLL_TIME:
            attempts += 1

            with self.client.get(
                f"/api/results/{upload_id}",
                name="/api/results/{upload_id}",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Poll failed: {response.status_code}")
                    return False, f"Poll failed: {response.status_code}"

                data = response.json()
                status = data.get("status")

                if status == "completed":
                    results = data.get("results", [])
                    response.success()
                    print(
                        f"Completed after {attempts} poll(s), "
                        f"found {len(results)} similar tracks"
                    )
                    return True, None

                elif status == "failed":
                    error = data.get("error", "Unknown error")
                    response.failure(f"Processing failed: {error}")
                    return False, f"Processing failed: {error}"

                elif status == "processing":
                    # Mark as success for the poll request itself
                    response.success()
                    # Wait before next poll
                    time.sleep(self.POLL_INTERVAL)

                else:
                    response.failure(f"Unknown status: {status}")
                    return False, f"Unknown status: {status}"

        # Timeout reached
        print(f"Polling timeout after {self.MAX_POLL_TIME}s and {attempts} attempts")
        return False, f"Timeout after {self.MAX_POLL_TIME}s"
