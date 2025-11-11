from locust import HttpUser, task, between


class FullUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def full_workflow(self):
        # Step 1: Get upload parameters
        response = self.client.get("/upload", name="Get Upload Params")
        upload_data = response.json()
        upload_url = upload_data["upload_url"]
        upload_id = upload_data["upload_id"]

        # Step 2: Upload file to presigned URL
        with open("test_track.mp3", "rb") as f:
            self.client.put(upload_url, data=f, name="Upload File")

        # Step 3: Notify server of the upload
        self.client.post(f"/uploads/{upload_id}", name="Start Processing")

        # Step 4: Poll for results
        while True:
            result_response = self.client.get(
                f"/results/{upload_id}", name="Poll Results"
            )
            result_data = result_response.json()
            if result_data["status"] in ["completed", "failed"]:
                print(
                    f"Upload {upload_id} finished with status: {result_data['status']}"
                )
                break
