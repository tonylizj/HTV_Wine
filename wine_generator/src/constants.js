import firebase from 'firebase';
const config = {
    apiKey: "AIzaSyBBaJIkRtF3jgzFF7BC83G6nI9joZKxezg",
    authDomain: "doppelganger-app.firebaseapp.com",
    databaseURL: "https://doppelganger-app.firebaseio.com",
    projectId: "doppelganger-app",
    storageBucket: "doppelganger-app.appspot.com",
    messagingSenderId: "115334021919"
};
firebase.initializeApp(config);
export const googleProvider = new firebase.auth.GoogleAuthProvider();
export const firebaseAuth = firebase.auth;
export const db = firebase.firestore().settings({ timestampsInSnapshots: true });
