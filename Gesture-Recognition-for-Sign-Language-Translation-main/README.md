# Gesture-Recognition-for-Sign-Language-Translation
This program will use gesture detection to help identify common ASL gestures as well as alphabets, translating them into sentences. These will be converted to speech This application will be converted into a Android application for greater usability.


<br>

## Steps to run

1. Install all the required node modules by using `npm install` in the root working directory of the project.
2. Run the express server using the command `nodemon app.js`.
3. The default port is 3000 so open up `localhost:3000` on Google Chrome
4. Let the model load up before predicting the gestures


<br>

## File Descriptions

- `public/`: Files related to the webpage including the HTML/CSS files for page design aand JS scripts
  - `models/`: TensorflowJS models for the alphabet and gesture models 
 
- `index.js`: Nodejs base file to run the webpage


<br> 


