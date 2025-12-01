import { useState } from "react";
import "./App.css";

function App() {
  const [inputValue, setInputValue] = useState("");
  const [result, setResult] = useState("");

  const predictHandler = async () => {
    console.log(`Se introdujo: ${inputValue}`);

    try {
      const response = await fetch("http://127.0.0.1:5000/predecir", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texto: inputValue }),
      });

      if (!response.ok) throw new Error("Error del servidor");

      const data = await response.json();
      console.log(data);

      setResult(data.texto_predicho);
    } catch (error) {
      console.error(error);
      alert("Error al conectar con el backend");
    }
  };

  return (
    <div>
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="una oracion en minusculas y sin puntear"
      />
      <button onClick={predictHandler}>Magia</button>
      {result && <p>Resultado: {result}</p>}
    </div>
  );
}

export default App;
