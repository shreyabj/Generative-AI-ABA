import React from "react";
import ReactDOM from "react-dom/client";
import App2 from "./v2/App2";
import "./styles/global.css";
import "./styles/app.css";
import "./v2/v2.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App2 />
  </React.StrictMode>,
);
