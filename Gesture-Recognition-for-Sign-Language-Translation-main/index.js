var express = require('express');
const path = require('path');
var app = express();
const port = 3000

app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.use(express.static(__dirname + "/public"))


app.get('/sample', function (req, res) {
    // var model = loadModel();
    request('http://127.0.0.1:5000/translate_to_Konkani', function (error, response, body) {
        console.error('error:', error); // Print the error
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        console.log('body:', body); // Print the data received
        res.send(body); //Display the response on the website
      });    
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})

