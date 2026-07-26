import { PgnAnalysisPanel } from "./PgnAnalysisPanel.js";

function displayName(id: string): string {
  return id
    .split("-")
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

export function App(): React.JSX.Element {
  return (
    <main>
      <header className="product-header">
        <div>
          <span className="eyebrow">Completed-game research tool</span>
          <h1>DrawbackGuesser</h1>
        </div>
        <p>
          Analyze a completed PGN locally. The symbolic engine controls hard
          eliminations; neural models can rank only surviving hypotheses.
        </p>
      </header>
      <aside className="safety-note">
        This staging application does not connect to external chess sites or
        provide live competitive assistance.
      </aside>
      <PgnAnalysisPanel nameFor={displayName} />
    </main>
  );
}
