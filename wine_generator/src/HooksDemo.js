import React, { useState, useEffect } from 'react';
import './App.css';

export default function HookDemo() {

  const [price, setPrice] = useInput('')
  const [category, setCategory] = useInput('');

  return (
    <form className="App-header">
      <h1>Enter Price Range and Category</h1>
      <label>
        Price
        <input
          type="text"
          onChange={setPrice}
          value={price}
          />
      </label>
      <label>
        Category
        <input
          type ="text"
          onChange={setCategory}
          value={category}
        />
      </label>
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
