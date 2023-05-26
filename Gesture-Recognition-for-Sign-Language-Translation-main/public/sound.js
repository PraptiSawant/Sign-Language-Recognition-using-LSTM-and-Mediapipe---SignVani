const speakBtn = document.getElementById("speak-btn");
const endpoint_url = "https://siya02-konakni-tts.hf.space/api/predict/";
const myHeaders = new Headers();
myHeaders.append("Content-Type", "application/json");

function speakText() { 
  const konInput = document.getElementById('input').textContent;
  const payload = {
"data": [
  konInput,
  "Female",
  0.667,
  1,
  0,
  1,
  1
]
};

  const requestOptions = {
    method: 'POST',
headers: myHeaders,
body: JSON.stringify(payload),
redirect: 'follow'
  };

  fetch(endpoint_url, requestOptions)
    .then(response => response.json())
.then(result => {
    console.log(result);
    const audioData = result.data[0].substring(22);  
    const audioBlob = b64toBlob(audioData, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    const audioElement = new Audio(audioUrl);
    audioElement.play();
  })

    .catch(error => console.log('error', error));
}

function b64toBlob(base64, type) {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);

  for (let i = 0; i < len; ++i) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return new Blob([bytes], { type });
}

speakBtn.addEventListener("click", speakText);