# Instrucciones para Ejecutar el Pipeline Completo

## Resumen

✓ **SÍ existe un script que ejecuta todo el pipeline completo**: `scripts/run_complete_pipeline.py`
✓ **Creado job maestro de Slurm**: `slurm/00_run_complete_pipeline.job`
✓ **Scripts verificados**: El dry-run funciona correctamente
✓ **Datos verificados**: Los datasets existen en las ubicaciones correctas
✓ **Contenedor verificado**: `qwen_pipeline_v2.sif` está disponible

## Estado de Verificación

### 1. Compatibilidad del Pipeline ✓
```bash
# Ejecutado con éxito:
python scripts/run_complete_pipeline.py --dry_run --seed 42 --data_root "/mnt/fast/nobackup/users/gb0048/opro2/data"
```
Resultado: ✓ Todas las 7 etapas pasan el dry-run

### 2. Dependencias ✓
- Las librerías (PyTorch, Transformers, PEFT) están en el contenedor Singularity
- No se pueden verificar desde el nodo actual (datamove1) pero los jobs existentes las usan

### 3. Datos ✓
Rutas ajustadas a los datos existentes:
- ✓ Manifest: `/mnt/fast/nobackup/users/gb0048/opro2/data/processed/conditions_final/conditions_manifest_split.parquet`
- ✓ Train: `/mnt/fast/nobackup/users/gb0048/opro2/data/processed/experimental_variants/train_metadata.csv`
- ✓ Test: `/mnt/fast/nobackup/users/gb0048/opro2/data/processed/experimental_variants/test_metadata.csv`
- ✓ Dev: `/mnt/fast/nobackup/users/gb0048/opro2/data/processed/experimental_variants/dev_metadata.csv`

### 4. Contenedor ✓
```bash
ls -lh /mnt/fast/nobackup/users/gb0048/opro2/qwen_pipeline_v2.sif
# Enlace simbólico válido → /mnt/fast/nobackup/users/gb0048/opro/qwen_pipeline_v2.sif
```

## Cómo Enviar el Job al Cluster

### Opción 1: Job Maestro (Recomendado - 30 horas continuas)

Ejecuta todas las 7 etapas secuencialmente en un solo job:

```bash
# Desde un nodo con acceso a Slurm (NO desde datamove1):
cd /mnt/fast/nobackup/users/gb0048/opro2_clean
sbatch slurm/00_run_complete_pipeline.job 42
```

**Características:**
- **Tiempo total**: ~30 horas
- **GPU**: 1x RTX 3090
- **Memoria**: 64GB
- **Partición**: 3090
- **Seed**: 42 (o el que especifiques como argumento)

**Logs:**
- Output: `logs/00_complete_pipeline_<JOBID>.out`
- Errors: `logs/00_complete_pipeline_<JOBID>.err`

**Resultados:**
- Directorio: `results/complete_pipeline_seed42/`

### Opción 2: Jobs Individuales (7 jobs separados)

Si prefieres ejecutar las etapas por separado:

```bash
cd /mnt/fast/nobackup/users/gb0048/opro2_clean

# Etapa 2: LoRA Fine-tuning (12h)
sbatch slurm/01_finetune_lora.job 42

# Etapa 4: OPRO sobre modelo BASE (4h)
sbatch slurm/02_opro_base.job 42

# Etapa 5: OPRO sobre modelo LoRA (2.5h)
sbatch slurm/03_opro_lora.job 42

# Etapa 3a: Evaluación BASE (2h)
sbatch slurm/04_eval_base.job

# Etapa 3b: Evaluación LoRA (2h)
sbatch slurm/05_eval_lora.job 42

# Etapa 6: Evaluación BASE + OPRO (2h)
sbatch slurm/06_eval_base_opro.job 42

# Etapa 7: Evaluación LoRA + OPRO (2h)
sbatch slurm/07_eval_lora_opro.job 42
```

**Nota**: Los jobs individuales requieren ejecutarse en orden de dependencias.

## Verificar Estado del Job

```bash
# Ver jobs en cola
squeue -u gb0048

# Ver detalles de un job específico
scontrol show job <JOBID>

# Ver logs en tiempo real
tail -f logs/00_complete_pipeline_<JOBID>.out
```

## Resultados Esperados

| Configuración | Balanced Accuracy | Mejora |
|---------------|-------------------|---------|
| BASE (baseline) | ~80% | - |
| BASE + OPRO | ~86.9% | +6.9% |
| LoRA (baseline) | ~88% | +8% |
| **LoRA + OPRO** | **~93.7%** | **+13.7%** ⭐ |

## Archivos Creados/Modificados

1. ✓ **Creado**: `slurm/00_run_complete_pipeline.job` - Job maestro de Slurm
2. ✓ **Modificado**: `scripts/run_complete_pipeline.py` - Ajustadas rutas de datos a `experimental_variants/`
3. ✓ **Creado**: Este archivo de documentación

## Nota Importante sobre el Nodo Actual

El nodo actual (`datamove1.surrey.ac.uk`) es un nodo de transferencia de datos y **NO tiene acceso a comandos de Slurm** (`sbatch`, `squeue`, etc.).

Para enviar jobs, necesitas conectarte a un nodo de login del cluster que tenga acceso a Slurm.

## Próximos Pasos

1. Conéctate a un nodo con acceso a Slurm
2. Ejecuta: `sbatch slurm/00_run_complete_pipeline.job 42`
3. Monitorea con: `squeue -u gb0048`
4. Revisa logs en: `logs/00_complete_pipeline_*.out`
5. Revisa resultados en: `results/complete_pipeline_seed42/`

---

**Última actualización**: 2025-12-14
**Estado**: ✓ Listo para ejecutar
