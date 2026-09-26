import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';
const output = ts.transpileModule(fs.readFileSync('src/period-return.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { finiteNumber, periodReturn } = await import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`);
test('daily close-to-close, negative returns, and null values', () => {
  assert.equal(finiteNumber(null), null);
  assert.equal(finiteNumber(''), null);
  assert.equal(finiteNumber('0'), 0);
  assert.ok(Math.abs(periodReturn([{date:'2026-09-18',close:'10'}, {date:'2026-09-21',close:'11'}]) - 10) < 1e-9);
  assert.ok(Math.abs(periodReturn([{date:'2026-09-18',close:'10'}, {date:'2026-09-21',close:'9'}]) + 10) < 1e-9);
  for (const value of [null, '', 'NaN', '0', '-1']) {
    assert.equal(periodReturn([{date:'2026-09-18',close:value}, {date:'2026-09-21',close:'11'}]), null);
  }
});
test('weekly return uses previous close, including partial week', () => {
  const rows = [{date:'2026-09-18',close:'10'}, {date:'2026-09-24',open:'10.5',close:'11',is_complete:false}];
  assert.ok(Math.abs(periodReturn(rows) - 10) < 1e-9);
  assert.equal(periodReturn(rows.slice(0,1)), null);
  assert.equal(periodReturn([rows[1],rows[0]]), null);
});
