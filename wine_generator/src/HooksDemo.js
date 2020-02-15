import React, { useState, useEffect, Button } from 'react';
import './App.css';

export default function HookDemo() {

  const [category, setCategory] = useInput('');

  return (
    <form className="App-header">
      <h1>What Kind of Wine?</h1>
      <label className="field">
        <textarea
          className="field"
          type="text"
          onChange={setCategory}
          value={category}
          />
      </label>
      <button type='submit' className="submitButton"> 
        Find Wine </button>
    </form>
  );
}

function useInput(defaultValue) {
  const [value, setValue] = useState('');
  const handleValueChange = (event) => {
    setValue(event.target.value);
  };
  return [value, handleValueChange];
}
