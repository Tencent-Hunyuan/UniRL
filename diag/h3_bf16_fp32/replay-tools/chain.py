"""capture→等待CPU门禁→V1→V3→V2；私有进程组、10小时硬截止。"""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

T = Path(os.environ['CHECKUP_ROOT'])
SHM = Path(os.environ['CHECKUP_SHM'])
TRAIN = [0,1,2,3,6,7,8,9]
stop = False
deadline = float('inf')
active_token = None


def log(stage, event, elapsed=0, code=0, detail=''):
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    line = f'{now} stage={stage} event={event} elapsed_s={elapsed:.1f} exit_code={code} {detail}\n'
    with (T/'status.txt').open('a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    print(line, end='', flush=True)


def request_stop(signum, frame):
    global stop
    stop = True


def processes():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            pid = int(directory.name)
            stat = (directory/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] == 'Z':
                continue
            env = (directory/'environ').read_bytes().split(b'\0')
            token = next((x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CHECKUP_OWNER=')), None)
            cmd = (directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            result[pid] = {'pgid':int(stat[2]), 'token':token, 'cmd':cmd}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def cleanup(token):
    if not token:
        return
    for sig, seconds in [(signal.SIGTERM,20),(signal.SIGKILL,15)]:
        procs = processes()
        groups = {v['pgid'] for v in procs.values() if v['token'] == token}
        for group in groups:
            members = [v for v in procs.values() if v['pgid']==group]
            if not all(v['token']==token for v in members) or group == os.getpgrp():
                raise RuntimeError(f'refuse mixed/non-task process group {group}')
            try:
                os.killpg(group,sig)
            except ProcessLookupError:
                pass
        end = time.monotonic()+seconds
        while time.monotonic()<end:
            if not any(v['token']==token for v in processes().values()):
                return
            time.sleep(1)
    raise RuntimeError('task processes did not exit after cleanup')


def gpu_state():
    p = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
    return [dict(zip(['index','memory_mib','utilization_percent'],map(int,line.split(','))))
            for line in p.stdout.splitlines()]


def wait_free(stage):
    notified = False
    while True:
        if stop or time.time()>=deadline:
            raise TimeoutError('stopped or chain deadline reached')
        procs = processes()
        conversion = any('convert_unirl_lora_to_comfyui.py' in p['cmd'] for p in procs.values())
        heavy = any(any(x in p['cmd'] for x in ['ray::','raylet','torchrun','unirl.train_diffusion'])
                    for p in procs.values())
        busy = any(g['memory_mib'] > 0 for g in gpu_state() if g['index'] in TRAIN)
        mem = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))/1024**2
        if not conversion and not heavy and not busy and mem >= 60:
            return
        if not notified:
            log(stage,'WAIT_FREE',detail=f'conversion={conversion} heavy={heavy} gpu_busy={busy} available_gib={mem:.1f}')
            notified = True
        time.sleep(15)


def run_stage(stage, args, gpu=True):
    global active_token
    if gpu:
        wait_free(stage)
    if stop or time.time()>=deadline:
        return 124
    token = 'h3gc12-'+stage+'-'+uuid.uuid4().hex
    active_token = token
    env = os.environ.copy()
    env.update(CHECKUP_OWNER=token,CHECKUP_STAGE=stage,PYTHONDONTWRITEBYTECODE='1')
    if stage in ['capture','V1']:
        env['H3_DIAG_AG']='1'
    else:
        env.pop('H3_DIAG_AG',None)
    if not gpu:
        env['CUDA_VISIBLE_DEVICES']=''
    start = time.time()
    log(stage,'START')
    path = T/'logs'/f'{stage}.log'
    code = 1
    with path.open('w') as output:
        child = subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=T,env=env,
                                 stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        (T/'active-stage.json').write_text(json.dumps({'stage':stage,'pid':child.pid,'pgid':child.pid,'owner':token})+'\n')
        try:
            while child.poll() is None:
                if stop or time.time()>=deadline:
                    cleanup(token)
                    child.wait(timeout=10)
                    code = 124
                    break
                time.sleep(3)
            else:
                code = child.returncode
        finally:
            cleanup(token)
            active_token = None
    # Ray进程完全退出后再清掉本阶段的临时对象存储。
    stage_shm = SHM/stage
    if stage_shm.exists():
        shutil.rmtree(stage_shm)
    log(stage,'END' if code==0 else 'FAIL',time.time()-start,code,detail=f'log=logs/{stage}.log')
    return code


def summarize(stage):
    script = T/'summarize/summarize.py'
    if not script.is_file():
        log(stage,'SUMMARY_SKIPPED',detail='summarize/summarize.py not ready')
        return
    run_stage('summary-'+stage,[script,'--root',T,'--output',T/'results'],gpu=False)


def main():
    global deadline
    wait_free('capture')
    start = time.time()
    deadline = start+10*3600
    (T/'chain.json').write_text(json.dumps({'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),
        'capture_start_unix':start,'deadline_unix':deadline,'max_wall_seconds':36000},indent=2)+'\n')
    if run_stage('capture',[T/'tools/capture.py']) != 0 or not (T/'batch/DONE').exists():
        raise RuntimeError('capture failed; no valid common batch; see logs/capture.log')
    log('chain','WAIT_REPLAY_CPU_GATE')
    while not (T/'REPLAY_READY').exists():
        if stop or time.time()>=deadline:
            raise TimeoutError('deadline while waiting for replay gate')
        time.sleep(10)
    checks=json.loads((T/'results/cpu-checks-replay.json').read_text())
    assert checks['status']=='PASS'
    for variant,micro in [('V1',8),('V3',4),('V2',1)]:
        if stop or time.time()>=deadline:
            log('chain','BUDGET_STOP',code=124,detail=f'remaining_from={variant}')
            break
        code=run_stage(variant,[T/'tools/replay.py','--variant',variant,'--micro',micro])
        if code and variant=='V3' and not stop and time.time()<deadline:
            text=(T/'logs/V3.log').read_text(errors='replace').lower()
            if 'out of memory' in text:
                log('V3','OOM_RETRY_MICRO2',code=code)
                failed=T/'failed'/f'V3-m4-{int(time.time())}'
                failed.mkdir(parents=True)
                if (T/'gradients/V3').exists():
                    shutil.move(str(T/'gradients/V3'),str(failed/'gradients'))
                shutil.copy2(T/'logs/V3.log',failed/'V3-m4.log')
                code=run_stage('V3-m2',[T/'tools/replay.py','--variant','V3','--micro',2])
        summarize(variant)
    log('chain','COMPLETE')


signal.signal(signal.SIGTERM,request_stop)
signal.signal(signal.SIGINT,request_stop)
exit_code=0
try:
    main()
except TimeoutError as exc:
    exit_code=124
    log('chain','BUDGET_OR_USER_STOP',code=124,detail=str(exc))
except BaseException as exc:
    exit_code=1
    log('chain','FAIL',code=1,detail=f'{type(exc).__name__}: {exc}')
    import traceback
    traceback.print_exc()
finally:
    try:
        cleanup(active_token)
        final={'time':time.time(),'gpu':gpu_state(),'remaining_task_processes':
               [pid for pid,p in processes().items() if p['token'] and p['token'].startswith('h3gc12-')]}
        assert not final['remaining_task_processes']
        if SHM.exists():
            shutil.rmtree(SHM)
        final['remote_task_bytes']=int(subprocess.check_output(['du','-sb',str(T)]).split()[0])
        (T/'results/final-host-state-r3.json').write_text(json.dumps(final,indent=2)+'\n')
        log('chain','FINAL_HOST_STATE',code=exit_code,detail='results/final-host-state-r3.json')
    except BaseException as exc:
        log('chain','CLEANUP_FAILED',code=1,detail=str(exc))
        exit_code=1
sys.exit(exit_code)
