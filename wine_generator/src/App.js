import React from 'react';
import HookDemo from './HooksDemo';
import './App.css';

export default function App() {
  return (
    <div className="App">
      <header className="App-header">
        <ErrorBoundary fallback = {(error) => (
          <span>
            OOOPS
          </span>
        )}>
          <HookDemo />
        </ErrorBoundary>
      </header>
    </div>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { error: error };
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      const Fallback = this.props.fallback;
      return (
        <Fallback error={this.state.error}/>
      )
    }

    return this.props.children;
  }
}
