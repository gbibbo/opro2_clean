# OPRO2 (Surrey HPC) – Reglas operativas para Claude Code

## Contexto fijo
- Estoy trabajando en VS Code conectado por SSH a: datamove1
- Repo abierto en: /mnt/fast/nobackup/users/gb0048/opro2
- En datamove1 NO están los comandos de SLURM en el PATH.
- SLURM se opera desde el nodo submit/login: aisurrey-submit01.surrey.ac.uk
- No ejecutar cómputo pesado en submit; siempre enviar jobs con sbatch. (Buenas prácticas SLURM) :contentReference[oaicite:1]{index=1}

## Regla #1 (IMPORTANTE): Slurm siempre via wrapper
Para cualquier cosa de SLURM (squeue/sbatch/scancel/sacct/scontrol), ejecutar SIEMPRE vía:
  ./slurm/tools/on_submit.sh <comando> <args...>

Ejemplos:
- Ver cola:
  ./slurm/tools/on_submit.sh squeue -u gb0048    :contentReference[oaicite:2]{index=2}
- Enviar job:
  ./slurm/tools/on_submit.sh sbatch slurm/<script>.job  :contentReference[oaicite:3]{index=3}
- Cancelar:
  ./slurm/tools/on_submit.sh scancel <JOBID>
- Ver detalles del job (incluye Dependency):
  ./slurm/tools/on_submit.sh scontrol show job <JOBID>
- Ver histórico/estado final:
  ./slurm/tools/on_submit.sh sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS

## Regla #2: No adivinar, ejecutar y pegar salida real
Cuando el usuario pregunte:
- “¿Qué jobs hay en cola?” -> ejecutar squeue y responder con la salida.
- “¿Qué pasó con el job X?” -> ejecutar scontrol show job X y/o sacct -j X y responder con datos.

## Regla #3: Diagnóstico de DependencyNeverSatisfied
Si un job está PD con reason DependencyNeverSatisfied:
- Es una razón oficial de SLURM: “dependencia que nunca se va a satisfacer”. :contentReference[oaicite:4]{index=4}
- Pasos:
  1) ./slurm/tools/on_submit.sh scontrol show job <JOBID> | sed -n '1,120p'
  2) Identificar el campo Dependency=...
  3) Si la dependencia refiere a un JobID viejo/cancelado, normalmente hay que cancelar y re-enviar sin esa dependencia, o resetear la dependencia con scontrol (si aplica). :contentReference[oaicite:5]{index=5}

## Regla #4: Datasets y almacenamiento (evitar llenar HOME)
- Evitar guardar datasets grandes en /user/HS300/gb0048 (HOME). Se llena fácil y rompe SSH/known_hosts.
- Proyecto y outputs: /mnt/fast/nobackup/users/gb0048/opro2
- Datasets: preferir /mnt/fast/nobackup/users/gb0048/datasets (o dentro de /mnt/fast/nobackup/... si ya está organizado).
- En jobs, usar rutas absolutas en /mnt/fast/nobackup/... para inputs/outputs.

## Regla #5: Cómo responder a pedidos “operativos”
Si el usuario pide “corré X”:
1) Identificar el script .job o .sh correspondiente en slurm/
2) Proponer el comando exacto sbatch (vía wrapper)
3) Ejecutarlo (si el usuario lo pidió) y devolver JobID
4) Monitorear con squeue/sacct (vía wrapper)

## Nota sobre Claude Code
Este archivo define instrucciones de proyecto para que Claude trabaje consistentemente dentro del repo. :contentReference[oaicite:6]{index=6}
