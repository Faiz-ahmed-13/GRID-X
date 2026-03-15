


console.log("Crash risk JS loaded");

const DRIVERS = ["VER","NOR","LEC","HAM","RUS","PIA","SAI","ALO"];
const DEFAULT_PTS = {VER:250,NOR:220,LEC:210,HAM:190,RUS:170,PIA:150,SAI:140,ALO:130};

let crashGaugeChart = null;
let scGaugeChart = null;

const NAV_FIELDS = ["circuit","track_temp"];

document.addEventListener("keydown",function(e){

if(e.key!=="Enter") return;

const active=document.activeElement;
const id=active.id;

const staticIdx=NAV_FIELDS.indexOf(id);

if(staticIdx!==-1){
e.preventDefault();

if(staticIdx<NAV_FIELDS.length-1){
document.getElementById(NAV_FIELDS[staticIdx+1]).focus();
}else{
document.getElementById("grid_0").focus();
}

return;
}

const gridSelectMatch=id.match(/^grid_(\d+)$/);

if(gridSelectMatch){
e.preventDefault();
const i=parseInt(gridSelectMatch[1]);
document.getElementById(`pts_${i}`).focus();
return;
}

const gridPtsMatch=id.match(/^pts_(\d+)$/);

if(gridPtsMatch){
e.preventDefault();
const i=parseInt(gridPtsMatch[1]);

if(i<DRIVERS.length-1){
document.getElementById(`grid_${i+1}`).focus();
}else{
predictCrashRisk();
}

}

});

function buildGridList(){

const list=document.getElementById("grid-list");
list.innerHTML="";

DRIVERS.forEach((drv,i)=>{

const row=document.createElement("div");
row.className="grid-row";

row.innerHTML=`
<span class="grid-pos">P${i+1}</span>

<select id="grid_${i}" onchange="onDriverChange(${i})">
${DRIVERS.map(d=>`<option value="${d}" ${d===drv?"selected":""}>${d}</option>`).join("")}
</select>

<span class="grid-pts-label">PTS</span>

<input type="number"
id="pts_${i}"
class="air-input"
min="0"
max="500"
value="${DEFAULT_PTS[drv]}"
>
`;

list.appendChild(row);

});

refreshAllDropdowns();

}

function onDriverChange(i){

const drv=document.getElementById(`grid_${i}`).value;
document.getElementById(`pts_${i}`).value=DEFAULT_PTS[drv]||0;

refreshAllDropdowns();

}

function refreshAllDropdowns(){

const selected=DRIVERS.map((_,i)=>
document.getElementById(`grid_${i}`)?.value
);

DRIVERS.forEach((_,i)=>{

const sel=document.getElementById(`grid_${i}`);
if(!sel) return;

Array.from(sel.options).forEach(opt=>{

const takenElsewhere=
selected.some((s,j)=>j!==i&&s===opt.value);

opt.disabled=takenElsewhere;
opt.style.color=takenElsewhere?"#444":"#fff";

});

});

}

buildGridList();

function getPctColor(pct){

if(pct<15) return "#00ff87";
if(pct<30) return "#ffcc00";
return "#ff2a2a";

}

function makeGauge(canvasId,value,color,existingChart){

if(existingChart) existingChart.destroy();

const ctx=document.getElementById(canvasId).getContext("2d");

return new Chart(ctx,{
type:"doughnut",
data:{
datasets:[{
data:[value,100-value],
backgroundColor:[color,"rgba(255,255,255,0.05)"],
borderWidth:0,
circumference:180,
rotation:270
}]
},
options:{
responsive:true,
cutout:"75%",
plugins:{
legend:{display:false},
tooltip:{enabled:false}
}
}
});

}

function predictCrashRisk(){

console.log("Predict function triggered");

document.getElementById("results-section").style.display="none";
document.getElementById("error-msg").style.display="none";
document.getElementById("loading-msg").style.display="block";

const grid_positions=[];
const championship_standings={};

for(let i=0;i<DRIVERS.length;i++){

const drv=document.getElementById(`grid_${i}`).value;
const pts=parseFloat(document.getElementById(`pts_${i}`).value||"0");

grid_positions.push(drv);
championship_standings[drv]=pts;

}

const payload={
circuit:document.getElementById("circuit").value,
weather_wet:document.getElementById("weather_wet").checked,
track_temp:parseFloat(document.getElementById("track_temp").value||"25"),
grid_positions,
championship_standings
};

console.log("Payload:",payload);

const minDelay=new Promise(resolve=>setTimeout(resolve,1500));

Promise.all([
fetch("http://localhost:8000/crash-risk-predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(payload)
}).then(r=>r.json()),
minDelay
])

.then(([data])=>{

console.log("API response:",data);

document.getElementById("loading-msg").style.display="none";

if(!data){
document.getElementById("error-msg").style.display="block";
return;
}

const crashPct=Math.round((data.crash_probability||0)*100);
const scPct=Math.round((data.safety_car_probability||0)*100);

const crashColor=getPctColor(crashPct);
const scColor=getPctColor(scPct);

document.getElementById("crash-pct").textContent=crashPct+"%";
document.getElementById("crash-pct").style.color=crashColor;

document.getElementById("sc-pct").textContent=scPct+"%";
document.getElementById("sc-pct").style.color=scColor;

crashGaugeChart=makeGauge("crashGaugeChart",crashPct,crashColor,crashGaugeChart);
scGaugeChart=makeGauge("scGaugeChart",scPct,scColor,scGaugeChart);

const factorsList=document.getElementById("factors-list");
factorsList.innerHTML="";

(data.risk_factors||[]).forEach(item=>{

const isPositive=item.contribution&&item.contribution.startsWith("+");

const el=document.createElement("div");
el.className="factor-item";

el.innerHTML=`
<div class="factor-left">
<span class="factor-name">${item.factor.replace(/_/g," ")}</span>
<span class="factor-explanation">${item.explanation||""}</span>
</div>

<span class="factor-contribution" style="color:${isPositive?"#ff2a2a":"#0096ff"}">
${item.contribution}
</span>
`;

factorsList.appendChild(el);

});

const recsList=document.getElementById("recs-list");
recsList.innerHTML="";

(data.recommendations||[]).forEach(rec=>{

const el=document.createElement("div");
el.className="rec-item";
el.textContent=rec;

recsList.appendChild(el);

});

document.getElementById("results-section").style.display="block";

document.getElementById("results-section")
.scrollIntoView({behavior:"smooth"});

})

.catch(err=>{

console.log("Fetch error:",err);

document.getElementById("loading-msg").style.display="none";
document.getElementById("error-msg").style.display="block";

});

}