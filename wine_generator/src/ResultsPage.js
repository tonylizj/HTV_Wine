import React, { useState, useEffect, Button, Component, useContext } from 'react';
import { useRoutes, A} from 'hookrouter';
import { AppContext } from './App';
import './App.css';

const ResultsPage = () => {

  const {state, dispatch} = useContext(AppContext);
  const data = "";

  return (
    <div className="App">
      <h1> {state.country} {state.variety} {state.price} {state.description} </h1>
      <A href="/">
        <button className="submitButton">
          Get more wine
        </button>
      </A>
    </div>
  );
};

export default ResultsPage;
