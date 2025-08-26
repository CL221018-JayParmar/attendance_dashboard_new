const video = document.getElementById('video');
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

document.getElementById('start-btn').onclick = () => {
  let recorder = new MediaRecorder(video.srcObject);
  let chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = () => {
    let blob = new Blob(chunks, { type: 'video/webm' });
    uploadVideo(blob);
  };
  recorder.start();
  setTimeout(() => recorder.stop(), 5000); // 5-second clip
};

function uploadVideo(blob) {
  let form = new FormData();
  form.append('video', blob);
  fetch(window.location.href, { method: 'POST', body: form })
    .then(res => {
      if (!res.ok) return res.text().then(t => { throw new Error(t) });
      return res.json();
    })
    .then(data => alert(data.message))
    .catch(e => alert("Error uploading video: " + e.message));
}
