import React, { useState, useEffect, Button, useContext } from 'react';
import 'whatwg-fetch';
import ReactDOM from 'react-dom';
import { useRoutes, A } from 'hookrouter';
import './App.css';
import { AppContext } from './App';


export const InputPage = () => {

  const {state, dispatch} = useContext(AppContext);

  const [country, setCountry] = useInput('');
  const [variety, setVariety] = useInput('');
  const [price, setPrice] = useInput('');

  const handleSubmit = (formFields) => {
    formFields.preventDefault();
    var desc = '';
    const data = formFields.target.getElementsByTagName('textarea');
    window.fetch('https://jsonplaceholder.typicode.com/users')
    .then((response) => {
        desc = response.json();
      })
    dispatch({ type: 'update',
               country: data.country.value,
               variety: data.variety.value,
               price: data.price.value,
               description: desc });

  };

  return (
    <form className="App-header" onSubmit={handleSubmit}>
      <div>
        <h1>What Kind of Wine??</h1>
      </div>

      <div className="inputArea">
        <label className="singleForm">
          <div className="formTitle"> Country of Origin</div>
          <textarea
            className="field"
            name="country"
            type="text"
            onChange={setCountry}
            value={country}
            />
        </label>
        <label className="singleForm">
          <div className="formTitle"> Variety </div>
          <textarea
            className="field"
            name="variety"
            type="text"
            onChange={setVariety}
            value={variety}
            />
        </label>
        <label className="singleForm">
          <div className="formTitle"> Price ($) </div>
          <textarea
            className="field"
            name="price"
            type="text"
            onChange={setPrice}
            value={price}
            />
        </label>
        <button type='submit' className="submitButton"> Submit </button>
        <A href="/Results">
          <button className="submitButton">
            Find Wine
          </button>
        </A>
      </div>
    </form>
  );

};

function useInput(defaultValue) {
  const [value, setValue] = useState('');
  const handleValueChange = (event) => {
    setValue(event.target.value);
  };
  return [value, handleValueChange];
}

export default InputPage;
