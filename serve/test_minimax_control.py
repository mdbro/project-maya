import hashlib
import json
import pathlib
import tempfile
import types
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, serve,
                          minimax_profile_fnv64, MINIMAX_PROFILE_BYTES)

ROOT=pathlib.Path(__file__).resolve().parents[1]

class ControlEngine(MockEngine):
    def __init__(self,tok):
        super().__init__(tok,'ok',max_context=8192)
        self.info={'engine_kind':'mm3'}
        self.proc=types.SimpleNamespace(pid=1234,poll=lambda:None)
        self.calls=[];self.bad_digest=False
        self.profile=bytes([1,0,0,0]) * (MINIMAX_PROFILE_BYTES//4)
    def command(self,line,expect,timeout):
        self.calls.append((line,expect,timeout))
        parts=line.split()
        if parts[0]=='PROFILE':
            result={'action':parts[1]}
            if parts[1]=='snapshot': result['profile_hex']=self.profile.hex()
        elif parts[1] in ('inspect','snapshot','restore','backing-on','backing-off','replay','replay-services'): result={'ownership_complete':True}
        else:
            data=bytes.fromhex(parts[2])
            result={'profile_fnv64':'bad' if self.bad_digest else minimax_profile_fnv64(data),
                    'ownership_complete':True}
        return parts[0]+' '+json.dumps(result)

class MiniMaxControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok=ByteTokenizer();cls.engine=ControlEngine(tok)
        cls.svc=Service(cls.engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        cls.httpd=serve(cls.svc,port=0);cls.base=f'http://127.0.0.1:{cls.httpd.server_address[1]}'
    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown();cls.httpd.server_close()
    def post(self,path,body):
        req=urllib.request.Request(self.base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req) as r: return r.status,json.load(r)
        except urllib.error.HTTPError as e: return e.code,json.load(e)
    def test_profile_snapshot_reset(self):
        status,res=self.post('/v1/minimax/profile',{'action':'snapshot'})
        self.assertEqual(status,200);self.assertEqual(res['engine_pid'],1234)
        self.assertEqual(res['profile_sha256'],hashlib.sha256(self.engine.profile).hexdigest())
        self.assertEqual(bytes.fromhex(res['profile_hex']),self.engine.profile)
        self.assertEqual(self.post('/v1/minimax/profile',{'action':'reset'})[0],200)
    def test_placement_validation_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'profile';path.write_bytes(self.engine.profile)
            status,res=self.post('/v1/minimax/placement',{'action':'helper','profile_path':str(path)})
            self.assertEqual(status,200);self.assertEqual(res['requested_profile_sha256'],hashlib.sha256(self.engine.profile).hexdigest())
            self.assertEqual(res['applied_profile_sha256'],res['requested_profile_sha256'])
            self.assertTrue(self.engine.calls[-1][0].startswith('PLACEMENT helper '))
            self.engine.bad_digest=True
            try: self.assertEqual(self.post('/v1/minimax/placement',{'action':'all','profile_path':str(path)})[0],400)
            finally: self.engine.bad_digest=False
            n=len(self.engine.calls);path.write_bytes(bytes(MINIMAX_PROFILE_BYTES))
            self.assertEqual(self.post('/v1/minimax/placement',{'action':'helper','profile_path':str(path)})[0],400)
            self.assertEqual(len(self.engine.calls),n)
    def test_inspect_and_bad_commands(self):
        self.assertEqual(self.post('/v1/minimax/placement',{'action':'inspect'})[0],200)
        self.assertEqual(self.engine.calls[-1][0],'PLACEMENT inspect')
        self.assertEqual(self.post('/v1/minimax/profile',{'action':'unknown'})[0],400)
    def test_learned_state_controls_are_profile_free(self):
        for action in ('snapshot','restore','backing-on','backing-off'):
            self.assertEqual(self.post('/v1/minimax/placement',{'action':action})[0],200)
            self.assertEqual(self.engine.calls[-1][0],'PLACEMENT '+action)
            n=len(self.engine.calls)
            self.assertEqual(self.post('/v1/minimax/placement',{'action':action,'profile_path':'ignored'})[0],400)
            self.assertEqual(len(self.engine.calls),n)

    def test_replay_control_validates_file_before_submission(self):
        self.assertEqual(self.post('/v1/minimax/placement',{'action':'replay-services'})[0],200)
        n=len(self.engine.calls)
        self.assertEqual(self.post('/v1/minimax/placement',{'action':'replay','workload_path':'relative'})[0],400)
        self.assertEqual(self.post('/v1/minimax/placement',{'action':'inspect','workload_path':'/tmp/unused'})[0],400)
        self.assertEqual(len(self.engine.calls),n)
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'workload';path.write_text('complete parser validation is native')
            self.assertEqual(self.post('/v1/minimax/placement',{'action':'replay','workload_path':str(path)})[0],200)
            self.assertEqual(self.engine.calls[-1][0],'PLACEMENT replay '+str(path))
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'profile';path.write_bytes(self.engine.profile)
            self.assertEqual(self.post('/v1/minimax/placement',{'action':'helper-first','profile_path':str(path)})[0],200)
            self.assertTrue(self.engine.calls[-1][0].startswith('PLACEMENT helper-first '))
    def test_state_test_gate_and_sampling_plumbing(self):
        engine=self.engine
        tune={'spec':8,'spec_policy':'fast_hsd','spec_rho':.2,'state_test_reject_first':1}
        req={'strata_tune':tune,'temperature':1,'top_p':.95,'top_k':0,'min_p':0}
        old=engine.info
        try:
            engine.info={'engine_kind':'mm3','dspark':1,'state_test_control':0}
            with self.assertRaises(ValueError): self.svc.check_request(dict(req))
            engine.info={'engine_kind':'mm3','dspark':1,'state_test_control':1}
            self.svc.check_request(dict(req))
            self.assertIn('state_test_reject_first=1',StrataEngine.sampling_keys({'strata_tune':tune},'mm3',engine.info))
            for value in (True,2,'1'):
                bad={**tune,'state_test_reject_first':value}
                with self.assertRaises(ValueError): self.svc.check_request({**req,'strata_tune':bad})
            with self.assertRaises(ValueError):
                self.svc.check_request({**req,'strata_tune':{**tune,'spec_policy':'existing'}})
        finally: engine.info=old

    def test_policy_diagnostics_history(self):
        eng=self.engine
        keys=('spec_policy','spec_rho','spec_ell','fresh_context','primary_policy','primary_horizon','primary_max_exchanges',
              'primary_service_ms','primary_service_effective_ms','primary_h2d_ms','primary_d2h_ms',
              'primary_interference_ms','primary_margin_ms')
        eng.stat_history=None
        StrataEngine._parse_stat(eng, 'STAT spec_policy=lenience spec_rho=-1 spec_ell=0.4 fresh_context=1 '
            'primary_policy=cost primary_horizon=20 primary_max_exchanges=8 primary_service_ms=0 primary_service_effective_ms=1 '
            'primary_h2d_ms=2 primary_d2h_ms=2 primary_interference_ms=0 primary_margin_ms=0')
        tiers=self.svc.metrics()['tiers']
        for key in keys:
            self.assertEqual(tiers['history'][key], [tiers['now'][key]])

    def test_full_ordinary_history_projection(self):
        eng=self.engine
        eng.stat_history=None
        def stat(count, ms, complete, final):
            StrataEngine._parse_stat(eng, f'STAT emitted={count} elapsed_ms={ms} request_seq=7 round_complete={complete} generation_final={final}')
        stat(0,0,1,0)
        for i in range(1,257):
            if i%8==0:stat(i,i*10-1,0,0)
            stat(i,i*10,1,0)
        stat(256,2561,1,1)
        h=self.svc.metrics()['tiers']['history']
        self.assertEqual(len(h['emitted']),290)
        self.assertEqual((h['emitted'][0],h['elapsed_ms'][0],h['round_complete'][0]),(0,0,1))
        self.assertEqual((h['emitted'][-1],h['generation_final'][-1]),(256,1))
        from tools.mm3 import native_screen
        joined=native_screen.summarize_stats({},self.svc.metrics(),{'completion_tokens':256})
        self.assertEqual(joined['generation_interval']['start_emitted'],32)
        self.assertEqual(joined['generation_interval']['end_emitted'],256)

if __name__=='__main__': unittest.main()
