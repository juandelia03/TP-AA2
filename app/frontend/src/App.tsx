import { useState } from "react";
import "./App.css";

function App() {
  const [inputValue, setInputValue] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const predictHandler = async () => {
    console.log(`Se introdujo: ${inputValue}`);
    setLoading(true);

    try {
      const response = await fetch(
        "https://juandelia03-t-aa-2.hf.space/predecir",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texto: inputValue }),
        }
      );

      if (!response.ok) throw new Error("Error del servidor");

      const data = await response.json();
      console.log(data);

      setResult(data.texto_predicho);
    } catch (error) {
      console.error(error);
      alert("Error al conectar con el backend");
    } finally {
      setLoading(false);
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
      <button onClick={predictHandler} disabled={loading}>
        {loading ? "Cargando..." : "Magia"}
      </button>
      {loading && <div className="spinner"></div>}
      {result && <p>Resultado: {result}</p>}
    </div>
  );
}

export default App;
