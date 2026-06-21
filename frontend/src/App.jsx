import { useState, useCallback, useEffect, useRef } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────
const ROWS = 20;
const COLS = 20;
const API  = "https://ai-disaster-evacuation-system.onrender.com";

const CELL = { FREE:0, WALL:1, START:2, HAZARD:3, CONGESTED:4, GOAL:5, PATH:6, VISITED:7, SHELTER:8, POPULATION:9 };

const CELL_COLORS = {
  0:"#0a1208", 1:"#1e2030", 2:"#00ff88", 3:"#ff3b3b",
  4:"#ff8c00", 5:"#00cfff", 6:"#ffe566", 7:"#0d2010",
  8:"#bf5fff", 9:"#ff69b4",
};

const ALGORITHMS = [
  { id:"astar",     label:"A*",         tag:"OPTIMAL",    color:"#00ff88" },
  { id:"idastar",   label:"IDA*",       tag:"MEMORY ↓",   color:"#00cfff" },
  { id:"bfs",       label:"BFS",        tag:"UNWEIGHTED", color:"#ffe566" },
  { id:"dfs",       label:"DFS",        tag:"EXPLORE",    color:"#ff8c00" },
  { id:"bestfirst", label:"Best-First", tag:"FAST",       color:"#ff69b4" },
  { id:"minimax",   label:"Minimax",    tag:"DYNAMIC",    color:"#ff3b3b" },
];

const DANGER_OVERLAY = { 0:"transparent", 1:"rgba(0,255,136,0.08)", 2:"rgba(255,59,59,0.22)" };

function makeGrid() { return Array.from({length:ROWS},()=>Array(COLS).fill(0)); }

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [grid, setGrid]             = useState(makeGrid);
  const [drawMode, setDrawMode]     = useState("start");
  const [start, setStart]           = useState(null);
  const [goal, setGoal]             = useState(null);
  const [population, setPopulation] = useState([]);
  const [shelters, setShelters]     = useState([]);
  const [algorithm, setAlgorithm]   = useState("astar");
  const [heuristic, setHeuristic]   = useState("euclidean");
  const [result, setResult]         = useState(null);
  const [gaResult, setGaResult]     = useState(null);
  const [mlResult, setMlResult]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [mlLoading, setMlLoading]   = useState(false);
  const [tab, setTab]               = useState("routing");
  const [numShelters, setNumShelters] = useState(3);
  const [generations, setGenerations] = useState(50);
  const [showDanger, setShowDanger]   = useState(false);
  const [dangerMap, setDangerMap]     = useState({});
  const isMouseDown = useRef(false);

  // ─── Display grid ────────────────────────────────────────────────────────
  const displayGrid = useCallback(() => {
    const g = grid.map(r=>[...r]);
    if (result) {
      result.visited?.forEach(([r,c])=>{ if(g[r][c]===0||g[r][c]===4) g[r][c]=CELL.VISITED; });
      result.path?.forEach(([r,c])=>{ if(g[r][c]!==CELL.START&&g[r][c]!==CELL.GOAL) g[r][c]=CELL.PATH; });
    }
    shelters.forEach(([r,c])=>{ g[r][c]=CELL.SHELTER; });
    population.forEach(([r,c])=>{ if(g[r][c]===0) g[r][c]=CELL.POPULATION; });
    if(start) g[start[0]][start[1]]=CELL.START;
    if(goal)  g[goal[0]][goal[1]]=CELL.GOAL;
    return g;
  }, [grid, result, start, goal, shelters, population]);

  const applyDraw = useCallback((r,c) => {
    setGrid(prev=>{
      const g=prev.map(row=>[...row]);
      if(drawMode==="wall")       g[r][c]=CELL.WALL;
      else if(drawMode==="hazard")    g[r][c]=CELL.HAZARD;
      else if(drawMode==="congested") g[r][c]=CELL.CONGESTED;
      else if(drawMode==="erase")     g[r][c]=CELL.FREE;
      return g;
    });
    if(drawMode==="start"){ setStart([r,c]); setResult(null); setMlResult(null); }
    if(drawMode==="goal"){  setGoal([r,c]);  setResult(null); setMlResult(null); }
    if(drawMode==="population") setPopulation(prev=>prev.some(p=>p[0]===r&&p[1]===c)?prev:[...prev,[r,c]]);
  },[drawMode]);

  useEffect(()=>{ window.addEventListener("mouseup",()=>{ isMouseDown.current=false; }); },[]);

  // ─── API calls ───────────────────────────────────────────────────────────
  const runPathfinding = async () => {
    if(!start||!goal) return alert("Set a start and goal first!");
    setLoading(true); setResult(null); setMlResult(null);
    const rawGrid = grid.map(r=>r.map(c=>c===CELL.START||c===CELL.GOAL?0:c));
    const hazardSources = [];
    for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) if(grid[r][c]===CELL.HAZARD) hazardSources.push([r,c]);
    try {
      const res = await fetch(`${API}/find-path`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({grid:rawGrid,start,goal,algorithm,heuristic,hazard_sources:hazardSources})});
      const data = await res.json();
      setResult(data);
    } catch { alert("Backend not running.\n\ncd backend && uvicorn server:app --reload"); }
    setLoading(false);
  };

  const runGeneticAlgorithm = async () => {
    if(!population.length) return alert("Place population points first!");
    setLoading(true); setShelters([]); setGaResult(null);
    try {
      const res = await fetch(`${API}/optimize-shelters`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({grid,population,num_shelters:numShelters,generations})});
      const data = await res.json();
      setShelters(data.shelters||[]); setGaResult(data);
    } catch { alert("Backend not running."); }
    setLoading(false);
  };

  const runMLAnalysis = async () => {
    if(!result?.path?.length) return alert("Run a pathfinding algorithm first!");
    setMlLoading(true);
    const rawGrid = grid.map(r=>r.map(c=>c===CELL.START||c===CELL.GOAL?0:c));
    try {
      const res = await fetch(`${API}/ml-analysis`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({grid:rawGrid,path:result.path,start,goal,
          population,shelters,num_clusters:Math.min(3,Math.max(1,population.length))})});
      const data = await res.json();
      setMlResult(data);
      // Build danger map for overlay
      if(data.knn?.danger_map_sample) {
        const dm={};
        data.knn.danger_map_sample.forEach(({pos,level})=>{ dm[`${pos[0]},${pos[1]}`]=level; });
        setDangerMap(dm);
      }
    } catch { alert("Backend not running."); }
    setMlLoading(false);
  };

  const generateRandom = async () => {
    let newGrid;
    try {
      const res = await fetch(`${API}/generate-grid?rows=${ROWS}&cols=${COLS}`);
      const data = await res.json();
      // Validate grid from backend
      if (data && data.grid && Array.isArray(data.grid) && data.grid.length === ROWS && Array.isArray(data.grid[0])) {
        newGrid = data.grid;
      } else {
        throw new Error("Invalid grid from backend");
      }
    } catch {
      // Fallback: build grid locally
      newGrid = makeGrid();
      for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++){
        const rnd=Math.random();
        if(rnd<0.15)newGrid[r][c]=1;
        else if(rnd<0.25)newGrid[r][c]=3;
        else if(rnd<0.30)newGrid[r][c]=4;
      }
      // Ensure corners are free
      newGrid[0][0]=0;
      newGrid[ROWS-1][COLS-1]=0;
    }

    // Safety check
    if (!newGrid || !Array.isArray(newGrid) || newGrid.length === 0) return;

    // Auto-place start at top-left free cell
    let newStart = null;
    for(let r=0; r<ROWS && !newStart; r++)
      for(let c=0; c<COLS && !newStart; c++)
        if(newGrid[r] && newGrid[r][c] === 0) newStart = [r, c];

    // Auto-place goal at bottom-right free cell
    let newGoal = null;
    for(let r=ROWS-1; r>=0 && !newGoal; r--)
      for(let c=COLS-1; c>=0 && !newGoal; c--)
        if(newGrid[r] && newGrid[r][c] === 0) newGoal = [r, c];

    // Auto-scatter population points
    const freeCells = [];
    for(let r=0;r<ROWS;r++)
      for(let c=0;c<COLS;c++)
        if(newGrid[r] && newGrid[r][c] === 0) freeCells.push([r,c]);

    const shuffled = [...freeCells].sort(()=>Math.random()-0.5);
    const newPop = shuffled.slice(0, Math.min(5, shuffled.length)).filter(([r,c])=>
      !(newStart && r===newStart[0] && c===newStart[1]) &&
      !(newGoal  && r===newGoal[0]  && c===newGoal[1])
    );

    setGrid(newGrid);
    setStart(newStart);
    setGoal(newGoal);
    setPopulation(newPop);
    setResult(null); setShelters([]); setGaResult(null); setMlResult(null); setDangerMap({});
  };

  const clearAll = () => {
    setGrid(makeGrid()); setStart(null); setGoal(null);
    setResult(null); setShelters([]); setPopulation([]); setGaResult(null); setMlResult(null); setDangerMap({});
  };

  const dg = displayGrid();

  return (
    <div style={S.app}>
      {/* ── Header ── */}
      <header style={S.header}>
        <div style={S.headerLeft}>
          <span style={S.logo}>⚠</span>
          <div>
            <h1 style={S.title}>EVACUATION ROUTING SYSTEM</h1>
            <p style={S.subtitle}>Pathfinding · ML Analysis · Shelter Optimization</p>
          </div>
        </div>
        <div style={S.tabs}>
          {[["routing","🗺 PATHFINDING"],["shelter","🏥 SHELTER OPT."],["ml","🤖 ML ANALYSIS"]].map(([t,l])=>(
            <button key={t} style={{...S.tab,...(tab===t?S.tabActive:{})}} onClick={()=>setTab(t)}>{l}</button>
          ))}
        </div>
      </header>

      <div style={S.body}>
        {/* ── Left Panel ── */}
        <aside style={S.panel}>

          {/* ROUTING TAB */}
          {tab==="routing" && <>
            <Sec title="ALGORITHM">
              <div style={S.algoGrid}>
                {ALGORITHMS.map(a=>(
                  <button key={a.id}
                    style={{...S.algoBtn,...(algorithm===a.id?{...S.algoBtnActive,borderColor:a.color,color:a.color}:{})}}
                    onClick={()=>{setAlgorithm(a.id);setResult(null);}}>
                    <span style={S.algoLabel}>{a.label}</span>
                    <span style={{...S.algoTag,color:a.color}}>{a.tag}</span>
                  </button>
                ))}
              </div>
            </Sec>
            {(algorithm==="astar"||algorithm==="idastar"||algorithm==="bestfirst")&&(
              <Sec title="HEURISTIC">
                <div style={S.row}>
                  {["euclidean","manhattan"].map(h=>(
                    <button key={h} style={{...S.hBtn,...(heuristic===h?S.hBtnActive:{})}} onClick={()=>setHeuristic(h)}>
                      {h.toUpperCase()}
                    </button>
                  ))}
                </div>
              </Sec>
            )}
            <DrawModes drawMode={drawMode} setDrawMode={setDrawMode}
              modes={[{id:"start",label:"▶ Start",color:"#00ff88"},{id:"goal",label:"⚑ Goal",color:"#00cfff"},
                {id:"wall",label:"■ Wall",color:"#333"},{id:"hazard",label:"🔥 Hazard",color:"#ff3b3b"},
                {id:"congested",label:"🚗 Congested",color:"#ff8c00"},{id:"erase",label:"✕ Erase",color:"#666"}]}/>
            <Sec>
              <button style={{...S.btn,...S.btnPrimary}} onClick={runPathfinding} disabled={loading}>
                {loading?"COMPUTING…":"▶ RUN ALGORITHM"}
              </button>
              <div style={S.row}>
                <button style={{...S.btn,...S.btnSecondary,flex:1}} onClick={generateRandom}>⟳ RANDOM</button>
                <button style={{...S.btn,...S.btnDanger,flex:1}} onClick={clearAll}>✕ CLEAR</button>
              </div>
            </Sec>
            {result&&(
              <div style={S.statsBox}>
                <p style={S.secTitle}>RESULTS</p>
                <StatRow label="Algorithm"      value={result.algorithm?.toUpperCase()}/>
                <StatRow label="Path length"    value={result.path_length??'—'}/>
                <StatRow label="Path cost"      value={result.cost??'No path'} accent={!result.cost}/>
                <StatRow label="Nodes explored" value={result.nodes_explored}/>
                {!result.cost&&<p style={{color:"#ff3b3b",fontSize:11,textAlign:"center",marginTop:6}}>⚠ No path found</p>}
              </div>
            )}
          </>}

          {/* SHELTER TAB */}
          {tab==="shelter" && <>
            <Sec title="GENETIC ALGORITHM">
              <p style={S.hint}>Place population points, then optimize shelter placement.</p>
            </Sec>
            <DrawModes drawMode={drawMode} setDrawMode={setDrawMode}
              modes={[{id:"population",label:"👤 Population",color:"#ff69b4"},{id:"wall",label:"■ Wall",color:"#333"},
                {id:"hazard",label:"🔥 Hazard",color:"#ff3b3b"},{id:"erase",label:"✕ Erase",color:"#666"}]}/>
            <Sec title="PARAMETERS">
              <label style={S.label}>Shelters: <b style={{color:"#bf5fff"}}>{numShelters}</b></label>
              <input type="range" min="1" max="6" value={numShelters} onChange={e=>setNumShelters(+e.target.value)} style={S.slider}/>
              <label style={S.label}>Generations: <b style={{color:"#bf5fff"}}>{generations}</b></label>
              <input type="range" min="10" max="200" step="10" value={generations} onChange={e=>setGenerations(+e.target.value)} style={S.slider}/>
            </Sec>
            <Sec>
              <button style={{...S.btn,...S.btnPurple}} onClick={runGeneticAlgorithm} disabled={loading}>
                {loading?"EVOLVING…":"🧬 RUN GENETIC ALGO"}
              </button>
              <div style={S.row}>
                <button style={{...S.btn,...S.btnSecondary,flex:1}} onClick={generateRandom}>⟳ RANDOM</button>
                <button style={{...S.btn,...S.btnDanger,flex:1}} onClick={clearAll}>✕ CLEAR</button>
              </div>
            </Sec>
            {gaResult&&(
              <div style={S.statsBox}>
                <p style={S.secTitle}>GA RESULTS</p>
                <StatRow label="Shelters placed" value={gaResult.shelters?.length}/>
                <StatRow label="Generations"     value={gaResult.generations}/>
                <StatRow label="Final fitness"   value={gaResult.final_fitness?.toFixed(4)}/>
              </div>
            )}
          </>}

          {/* ML TAB */}
          {tab==="ml" && <>
            <Sec title="ML MODELS">
              <p style={S.hint}>Run pathfinding first, then analyse the route with ML models.</p>
              <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:6}}>
                {[["KNN","Zone Danger Classification","#00ff88"],
                  ["K-Means","Population Clustering","#00cfff"],
                  ["Naive Bayes","Route Safety Prediction","#ffe566"],
                  ["Neural Net","Evacuation Time Estimate","#ff69b4"]
                ].map(([name,desc,color])=>(
                  <div key={name} style={{borderLeft:`2px solid ${color}`,paddingLeft:8,marginBottom:4}}>
                    <div style={{fontSize:12,color,fontWeight:700}}>{name}</div>
                    <div style={{fontSize:10,color:"#445"}}>{desc}</div>
                  </div>
                ))}
              </div>
            </Sec>
            <Sec>
              <button style={{...S.btn,...S.btnML}} onClick={runMLAnalysis} disabled={mlLoading||!result?.path?.length}>
                {mlLoading?"ANALYSING…":"🤖 RUN ML ANALYSIS"}
              </button>
              <label style={{...S.label,display:"flex",alignItems:"center",gap:6,cursor:"pointer"}}>
                <input type="checkbox" checked={showDanger} onChange={e=>setShowDanger(e.target.checked)} style={{accentColor:"#ff3b3b"}}/>
                Show KNN danger overlay
              </label>
            </Sec>
          </>}

          {/* Legend */}
          <Sec title="LEGEND">
            <div style={S.legend}>
              {[[0,"Free"],[1,"Wall"],[CELL.START,"Start"],[CELL.GOAL,"Goal"],
                [CELL.HAZARD,"Hazard"],[CELL.CONGESTED,"Congested"],
                [CELL.PATH,"Path"],[CELL.VISITED,"Explored"],
                [CELL.SHELTER,"Shelter"],[CELL.POPULATION,"Population"]
              ].map(([t,l])=>(
                <div key={t} style={S.legendItem}>
                  <div style={{...S.legendDot,background:CELL_COLORS[t],border:t===0?"1px solid #1a2a1a":"none"}}/>
                  <span style={S.legendLabel}>{l}</span>
                </div>
              ))}
            </div>
          </Sec>
        </aside>

        {/* ── Main Area ── */}
        <main style={S.main}>
          {tab==="ml" && mlResult ? (
            <div style={{width:"100%",height:"100%",overflowY:"auto",padding:"12px 16px",boxSizing:"border-box"}}>
              <MLResultsPanel ml={mlResult}/>
            </div>
          ) : (
            <>
              <div style={S.gridWrapper} onMouseLeave={()=>{isMouseDown.current=false;}}>
                <div style={{...S.grid,gridTemplateColumns:`repeat(${COLS},1fr)`}}>
                  {dg.map((row,r)=>row.map((cell,c)=>{
                    const dangerLevel = showDanger ? (dangerMap[`${r},${c}`]||0) : 0;
                    return (
                      <div key={`${r}-${c}`}
                        style={{
                          ...S.cell,
                          background:CELL_COLORS[cell]||CELL_COLORS[0],
                          boxShadow:cell===CELL.PATH?"0 0 4px #ffe566":cell===CELL.START?"0 0 8px #00ff88":
                                    cell===CELL.GOAL?"0 0 8px #00cfff":cell===CELL.SHELTER?"0 0 8px #bf5fff":"none",
                          outline: dangerLevel===2?"1px solid rgba(255,59,59,0.5)":dangerLevel===1?"1px solid rgba(255,140,0,0.3)":"none",
                          backgroundColor: dangerLevel>0 && cell===0
                            ? (dangerLevel===2?"#1a0505":"#0d0d00")
                            : CELL_COLORS[cell],
                        }}
                        onMouseDown={()=>{isMouseDown.current=true;applyDraw(r,c);}}
                        onMouseEnter={()=>{if(isMouseDown.current)applyDraw(r,c);}}
                      />
                    );
                  }))}
                </div>
              </div>
              <div style={S.infoBar}>
                <span>{ROWS}×{COLS} grid</span>
                <span>{start?`Start:(${start[0]},${start[1]})`:"No start"} · {goal?`Goal:(${goal[0]},${goal[1]})`:"No goal"}</span>
                {tab==="shelter"&&<span>Population: {population.length} pts</span>}
                {mlResult&&<span style={{color:"#00ff88"}}>ML: ✓ analysed</span>}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

// ─── ML Results Panel ─────────────────────────────────────────────────────────
function MLResultsPanel({ ml }) {
  return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,alignItems:"start"}}>

      {/* KNN */}
      <div style={{...S.mlCard,borderColor:"#00ff88"}}>
        <p style={{...S.mlCardTitle,color:"#00ff88"}}>KNN — ZONE DANGER</p>
        <div style={S.row}>
          <MLPill label="SAFE"     value={ml.knn?.zone_counts?.safe}     color="#00ff88"/>
          <MLPill label="MODERATE" value={ml.knn?.zone_counts?.moderate} color="#ff8c00"/>
          <MLPill label="DANGER"   value={ml.knn?.zone_counts?.danger}   color="#ff3b3b"/>
        </div>
        <StatRow label="Path danger score" value={ml.knn?.avg_path_danger?.toFixed(2)}
          accent={ml.knn?.avg_path_danger>1}/>
        <AccuracyBar label="Accuracy" value={ml.metrics?.knn?.accuracy} color="#00ff88"/>
        <ConfusionMatrix cm={ml.metrics?.knn?.confusion_matrix} classes={["SAFE","MOD","DANG"]} color="#00ff88"/>
      </div>

      {/* Naive Bayes */}
      {ml.naive_bayes&&(
        <div style={{...S.mlCard,borderColor:"#ffe566"}}>
          <p style={{...S.mlCardTitle,color:"#ffe566"}}>NAIVE BAYES — ROUTE SAFETY</p>
          <div style={{textAlign:"center",padding:"6px 0"}}>
            <span style={{fontSize:16,fontWeight:700,
              color:ml.naive_bayes.prediction===0?"#00ff88":"#ff3b3b"}}>
              {ml.naive_bayes.label}
            </span>
          </div>
          <div style={S.row}>
            <MLPill label="P(SAFE)"  value={`${(ml.naive_bayes.probability_safe*100).toFixed(0)}%`}  color="#00ff88"/>
            <MLPill label="P(RISKY)" value={`${(ml.naive_bayes.probability_risky*100).toFixed(0)}%`} color="#ff3b3b"/>
          </div>
          <StatRow label="Hazard cells on path" value={ml.naive_bayes.features?.hazard_cells}/>
          <StatRow label="Detour ratio"          value={ml.naive_bayes.features?.detour_ratio}/>
          <AccuracyBar label="Accuracy" value={ml.metrics?.naive_bayes?.accuracy} color="#ffe566"/>
          <ConfusionMatrix cm={ml.metrics?.naive_bayes?.confusion_matrix} classes={["SAFE","RISKY"]} color="#ffe566"/>
        </div>
      )}

      {/* Neural Network */}
      {ml.neural_network&&(
        <div style={{...S.mlCard,borderColor:"#ff69b4"}}>
          <p style={{...S.mlCardTitle,color:"#ff69b4"}}>NEURAL NET — TIME ESTIMATE</p>
          <div style={{textAlign:"center",padding:"6px 0"}}>
            <span style={{fontSize:22,fontWeight:700,color:"#ff69b4"}}>
              {ml.neural_network.estimated_minutes} min
            </span>
          </div>
          <StatRow label="R² score" value={ml.metrics?.neural_network?.r2}/>
          <StatRow label="RMSE"     value={ml.metrics?.neural_network?.rmse}/>
          <LossChart history={ml.metrics?.neural_network?.loss_history||[]}/>
        </div>
      )}

      {/* K-Means */}
      {ml.kmeans&&(
        <div style={{...S.mlCard,borderColor:"#00cfff"}}>
          <p style={{...S.mlCardTitle,color:"#00cfff"}}>K-MEANS — EVAC ZONES</p>
          <StatRow label="Clusters"         value={ml.kmeans.centroids?.length}/>
          <StatRow label="Silhouette score" value={ml.kmeans.silhouette_score}/>
          <StatRow label="Inertia"          value={ml.kmeans.inertia}/>
          {Object.entries(ml.kmeans.assignments||{}).map(([k,v])=>(
            <div key={k} style={{fontSize:10,color:"#778",marginTop:2}}>
              Zone {+k+1}: centroid ({v.centroid.map(n=>n.toFixed(0)).join(",")}) → shelter ({v.shelter.join(",")}) · {v.distance}u
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function Sec({ title, children }) {
  return (
    <div style={{marginBottom:10}}>
      {title&&<p style={S.secTitle}>{title}</p>}
      {children}
    </div>
  );
}

function DrawModes({ drawMode, setDrawMode, modes }) {
  return (
    <Sec title="DRAW MODE">
      <div style={S.drawGrid}>
        {modes.map(m=>(
          <button key={m.id}
            style={{...S.drawBtn,borderColor:drawMode===m.id?m.color:"#1a2a1a",
              color:drawMode===m.id?m.color:"#556",background:drawMode===m.id?"#0a0f0a":"transparent"}}
            onClick={()=>setDrawMode(m.id)}>{m.label}
          </button>
        ))}
      </div>
    </Sec>
  );
}

function StatRow({ label, value, accent }) {
  return (
    <div style={S.statRow}>
      <span style={S.statLabel}>{label}</span>
      <span style={{...S.statValue,color:accent?"#ff3b3b":"#00ff88"}}>{value}</span>
    </div>
  );
}

function MLPill({ label, value, color }) {
  return (
    <div style={{flex:1,textAlign:"center",border:`1px solid ${color}22`,borderRadius:3,padding:"4px 2px"}}>
      <div style={{fontSize:9,color:"#445",letterSpacing:1}}>{label}</div>
      <div style={{fontSize:13,fontWeight:700,color}}>{value}</div>
    </div>
  );
}

function AccuracyBar({ label, value, color }) {
  const pct = Math.min(100, value||0);
  return (
    <div style={{marginTop:6,marginBottom:4}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
        <span style={{fontSize:9,color:"#445"}}>{label}</span>
        <span style={{fontSize:10,color,fontWeight:700}}>{pct}%</span>
      </div>
      <div style={{background:"#111",borderRadius:2,height:6,overflow:"hidden"}}>
        <div style={{width:`${pct}%`,height:"100%",background:color,transition:"width .5s",borderRadius:2}}/>
      </div>
    </div>
  );
}

function ConfusionMatrix({ cm, classes, color }) {
  if(!cm||!cm.length) return null;
  return (
    <div style={{marginTop:6}}>
      <p style={{fontSize:9,color:"#556",letterSpacing:1,marginBottom:3}}>CONFUSION MATRIX</p>
      <div style={{display:"grid",gridTemplateColumns:`auto ${classes.map(()=>"1fr").join(" ")}`,gap:2,fontSize:9}}>
        <div/>
        {classes.map(c=><div key={c} style={{color,textAlign:"center",fontWeight:700}}>{c}</div>)}
        {cm.map((row,i)=>[
          <div key={`l${i}`} style={{color:"#778",paddingRight:4}}>{classes[i]}</div>,
          ...row.map((v,j)=>(
            <div key={j} style={{
              textAlign:"center",borderRadius:2,padding:"2px 0",fontWeight:i===j?700:400,
              background:i===j?`${color}22`:"#0a0f0a",color:i===j?color:"#667"
            }}>{v}</div>
          ))
        ])}
      </div>
    </div>
  );
}

function LossChart({ history }) {
  if(!history.length) return null;
  const max = Math.max(...history);
  const min = Math.min(...history);
  const range = max - min || 1;
  const W=200, H=50, pad=4;
  const points = history.map((v,i)=>{
    const x = pad + (i/(history.length-1||1))*(W-2*pad);
    const y = pad + (1-(v-min)/range)*(H-2*pad);
    return `${x},${y}`;
  }).join(" ");
  return (
    <div style={{marginTop:6}}>
      <p style={{fontSize:9,color:"#556",letterSpacing:1,marginBottom:3}}>TRAINING LOSS</p>
      <svg width={W} height={H} style={{background:"#080d12",borderRadius:2,border:"1px solid #111"}}>
        <polyline points={points} fill="none" stroke="#ff69b4" strokeWidth="1.5"/>
      </svg>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const S = {
  app:         {minHeight:"100vh",background:"#060b0f",color:"#c0c8c0",fontFamily:"'Courier New',monospace",display:"flex",flexDirection:"column",userSelect:"none"},
  header:      {display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 20px",borderBottom:"1px solid #0f1f0f",background:"#040809"},
  headerLeft:  {display:"flex",alignItems:"center",gap:12},
  logo:        {fontSize:30,filter:"drop-shadow(0 0 8px #ff3b3b)"},
  title:       {margin:0,fontSize:16,letterSpacing:3,color:"#00ff88",fontWeight:700},
  subtitle:    {margin:0,fontSize:9,color:"#334",letterSpacing:2},
  tabs:        {display:"flex",gap:6},
  tab:         {background:"transparent",border:"1px solid #1a2a1a",color:"#445",padding:"5px 12px",cursor:"pointer",fontSize:10,letterSpacing:2,borderRadius:2,fontFamily:"inherit"},
  tabActive:   {borderColor:"#00ff88",color:"#00ff88",background:"#001a08"},
  body:        {display:"flex",flex:1,overflow:"hidden"},
  panel:       {width:256,background:"#040809",borderRight:"1px solid #0a1a0a",overflowY:"auto",padding:10,display:"flex",flexDirection:"column",gap:2},
  secTitle:    {fontSize:9,letterSpacing:3,color:"#334",margin:"0 0 6px",fontWeight:700},
  algoGrid:    {display:"grid",gridTemplateColumns:"1fr 1fr",gap:5},
  algoBtn:     {background:"transparent",border:"1px solid #1a2a1a",color:"#445",padding:"5px 7px",cursor:"pointer",borderRadius:2,textAlign:"left",display:"flex",flexDirection:"column",gap:2,transition:"all .12s",fontFamily:"inherit"},
  algoBtnActive:{background:"#050f05"},
  algoLabel:   {fontSize:12,fontWeight:700},
  algoTag:     {fontSize:9,letterSpacing:1},
  row:         {display:"flex",gap:5},
  hBtn:        {flex:1,background:"transparent",border:"1px solid #1a2a1a",color:"#445",padding:"4px",cursor:"pointer",fontSize:9,letterSpacing:1,borderRadius:2,fontFamily:"inherit"},
  hBtnActive:  {borderColor:"#ffe566",color:"#ffe566",background:"#1a1500"},
  drawGrid:    {display:"grid",gridTemplateColumns:"1fr 1fr",gap:4},
  drawBtn:     {background:"transparent",border:"1px solid #1a2a1a",padding:"4px 6px",cursor:"pointer",fontSize:10,borderRadius:2,textAlign:"left",transition:"all .12s",fontFamily:"inherit"},
  btn:         {width:"100%",padding:"8px",cursor:"pointer",fontSize:10,letterSpacing:2,fontWeight:700,border:"none",borderRadius:2,marginBottom:5,fontFamily:"inherit"},
  btnPrimary:  {background:"#00ff88",color:"#001a08"},
  btnPurple:   {background:"#bf5fff",color:"#0d001a"},
  btnML:       {background:"#1a0a2a",color:"#cf8fff",border:"1px solid #bf5fff"},
  btnSecondary:{background:"#0a1a1a",color:"#00cfff",border:"1px solid #00cfff"},
  btnDanger:   {background:"#1a0808",color:"#ff3b3b",border:"1px solid #ff3b3b"},
  statsBox:    {background:"#070c07",border:"1px solid #0f1f0f",borderRadius:3,padding:10,marginTop:3},
  statRow:     {display:"flex",justifyContent:"space-between",padding:"2px 0",borderBottom:"1px solid #0a150a"},
  statLabel:   {fontSize:10,color:"#334"},
  statValue:   {fontSize:11,fontWeight:700},
  hint:        {fontSize:10,color:"#334",lineHeight:1.5,marginBottom:4},
  label:       {display:"block",fontSize:10,color:"#334",marginBottom:3,letterSpacing:1},
  slider:      {width:"100%",marginBottom:8,accentColor:"#bf5fff"},
  legend:      {display:"grid",gridTemplateColumns:"1fr 1fr",gap:3},
  legendItem:  {display:"flex",alignItems:"center",gap:5},
  legendDot:   {width:9,height:9,borderRadius:2,flexShrink:0},
  legendLabel: {fontSize:9,color:"#334"},
  mlCard:      {background:"#070c07",border:"1px solid",borderRadius:3,padding:10},
  mlCardTitle: {fontSize:9,letterSpacing:2,fontWeight:700,marginBottom:6},
  main:        {flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:12},
  gridWrapper: {border:"1px solid #0a1a0a",boxShadow:"0 0 40px #001a0818",borderRadius:3,overflow:"hidden"},
  grid:        {display:"grid",gap:1,background:"#000a04",padding:1,cursor:"crosshair"},
  cell:        {width:27,height:27,borderRadius:1,transition:"background .06s",cursor:"crosshair"},
  infoBar:     {marginTop:8,display:"flex",gap:20,fontSize:9,color:"#223",letterSpacing:1},
};
