import React, { Component, useState, useReducer } from 'react';
import { useRoutes } from 'hookrouter';
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";
import InputPage from './InputPage';
import ResultsPage from './ResultsPage';
import './App.css';

const initialState = {
  country: '',
  variety: '',
  price: '',
  description: '',
};

export const AppContext = React.createContext(initialState);

function reducer(state, action) {
    switch (action.type) {
        case 'update':
            return {
                country: action.country,
                variety: action.variety,
                price: action.price,
            };
        default:
            return initialState;
    }
}

const routes = {
  "/": () => <InputPage />,
  "/Results": () => <ResultsPage />
}

export default function App() {

  const routeResult = useRoutes(routes);
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <div className="App">
      <AppContext.Provider value={{state, dispatch}}>
        {routeResult}
      </AppContext.Provider>
    </div>
  );
}
