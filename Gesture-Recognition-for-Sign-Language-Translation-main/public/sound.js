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
    const audioElement = new Audio(result.data[0]);
    audioElement.play();
     
  })

    .catch(error => console.log('error', error));
}
  

speakBtn.addEventListener("click", speakText);
