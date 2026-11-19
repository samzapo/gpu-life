import type { UnifiedEngineRefs } from '../engineTypes';
import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import structsShader from '../shaders/structs.wgsl?raw';
import projectionShader from '../shaders/projection.wgsl?raw';
import bioChemShader from '../shaders/bioChem.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import oceanSimHelperShader from '../shaders/oceanSim.wgsl?raw';
import constructShader from './construct.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import bioStepShader from './bioStep.wgsl?raw';
import simShader from './sim.wgsl?raw';

interface LinkedWorldState {
  headsBuffer: GPUBuffer;
  headsInitBuffer: GPUBuffer;
  linkedListBuffer: GPUBuffer;
  constructBindGroups: [GPUBindGroup, GPUBindGroup];
  valenceBindGroups: [GPUBindGroup, GPUBindGroup];
  bioBindGroups: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
}

const workgroupSize = 128;

let constructPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let bioPipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;

let worldState: LinkedWorldState | undefined;

function destroyState() {
  if (!worldState) return;
  worldState.headsBuffer.destroy();
  worldState.headsInitBuffer.destroy();
  worldState.linkedListBuffer.destroy();
  worldState = undefined;
}

export function setup(device: GPUDevice) {
  setupTimestamp(device, 'construct');
  constructPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: `${structsShader}\n${constructShader}` }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'llValence');
  valencePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${valenceShader}`,
      }),
      entryPoint: 'main',
    },
  });

  bioPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${bioStepShader}`,
      }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'linkedSim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${oceanSimHelperShader}\n${simShader}`,
      }),
      entryPoint: 'main',
    },
  });
}

export function start(device: GPUDevice, refs: UnifiedEngineRefs) {
  if (!constructPipeline || !valencePipeline || !bioPipeline || !simPipeline) {
    return;
  }

  destroyState();

  const { particleAmt, cellAmt } = refs;
  const headsLen = cellAmt + 1;
  const headsBuffer = device.createBuffer({
    size: headsLen * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'llHeadsBuffer',
  });
  const headsInitBuffer = device.createBuffer({
    size: headsLen * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'llHeadsInitBuffer',
  });
  const headsInitData = new Uint32Array(headsLen);
  headsInitData.fill(0xffffffff);
  headsInitData[0] = 0;
  device.queue.writeBuffer(headsInitBuffer, 0, headsInitData);

  // ListParticle: pos(3)+colour(1)+orient(3)+idx+next+phase+bioSpecies+bioEnergy = 12 scalars (48 bytes)
  const linkedListBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'llLinkedListBuffer',
  });

  const constructGroups: GPUBindGroup[] = [];
  const valenceGroups: GPUBindGroup[] = [];
  const bioGroups: GPUBindGroup[] = [];
  const simGroups: GPUBindGroup[] = [];

  for (let i = 0; i < 2; i++) {
    constructGroups.push(
      device.createBindGroup({
        layout: constructPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.simBuffer } },
          { binding: 1, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 2, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 3, resource: { buffer: headsBuffer } },
          { binding: 4, resource: { buffer: linkedListBuffer } },
        ],
      }),
    );

    valenceGroups.push(
      device.createBindGroup({
        layout: valencePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: refs.simBuffer } },
          { binding: 2, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 3, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 4, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 5, resource: { buffer: refs.valenceBuffer } },
          { binding: 6, resource: { buffer: headsBuffer } },
          { binding: 7, resource: { buffer: linkedListBuffer } },
        ],
      }),
    );

    bioGroups.push(
      device.createBindGroup({
        layout: bioPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.bioUniformBuffer } },
          { binding: 1, resource: refs.oceanTexView },
          { binding: 2, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 3, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 4, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 5, resource: { buffer: refs.lifeStateBuffers[1 - i] } },
          { binding: 6, resource: { buffer: refs.valenceBuffer } },
          { binding: 7, resource: { buffer: refs.bioForceBuffer } },
          { binding: 8, resource: { buffer: headsBuffer } },
          { binding: 9, resource: { buffer: linkedListBuffer } },
          { binding: 10, resource: refs.windTexView },
          { binding: 11, resource: refs.tempTexView },
        ],
      }),
    );

    simGroups.push(
      device.createBindGroup({
        layout: simPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.uniformBuffer } },
          { binding: 1, resource: { buffer: refs.simBuffer } },
          { binding: 2, resource: { buffer: refs.oceanSimBuffer } },
          { binding: 3, resource: refs.windTexView },
          { binding: 4, resource: refs.oceanTexView },
          { binding: 5, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 6, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 7, resource: { buffer: refs.particleBuffers[1 - i] } },
          { binding: 8, resource: { buffer: refs.lifeStateBuffers[1 - i] } },
          { binding: 9, resource: { buffer: refs.valenceBuffer } },
          { binding: 10, resource: { buffer: refs.bioForceBuffer } },
          { binding: 11, resource: { buffer: headsBuffer } },
          { binding: 12, resource: { buffer: linkedListBuffer } },
          { binding: 13, resource: refs.tempTexView },
        ],
      }),
    );
  }

  worldState = {
    headsBuffer,
    headsInitBuffer,
    linkedListBuffer,
    constructBindGroups: [constructGroups[0], constructGroups[1]],
    valenceBindGroups: [valenceGroups[0], valenceGroups[1]],
    bioBindGroups: [bioGroups[0], bioGroups[1]],
    simBindGroups: [simGroups[0], simGroups[1]],
  };
}

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
) {
  const s = worldState;
  if (!s || !constructPipeline || !valencePipeline || !bioPipeline || !simPipeline) {
    return;
  }

  const wgCount = Math.ceil(particleAmt / workgroupSize);

  commandEncoder.copyBufferToBuffer(s.headsInitBuffer, s.headsBuffer);

  const constructPass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'construct'),
  );
  constructPass.setPipeline(constructPipeline);
  constructPass.setBindGroup(0, s.constructBindGroups[alternate]);
  constructPass.dispatchWorkgroups(wgCount);
  constructPass.end();
  resolveTimestamp(commandEncoder, 'construct');

  const valencePass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'llValence'),
  );
  valencePass.setPipeline(valencePipeline);
  valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
  valencePass.dispatchWorkgroups(wgCount);
  valencePass.end();
  resolveTimestamp(commandEncoder, 'llValence');

  const bioPass = commandEncoder.beginComputePass({});
  bioPass.setPipeline(bioPipeline);
  bioPass.setBindGroup(0, s.bioBindGroups[alternate]);
  bioPass.dispatchWorkgroups(wgCount);
  bioPass.end();

  const simPass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'linkedSim'),
  );
  simPass.setPipeline(simPipeline);
  simPass.setBindGroup(0, s.simBindGroups[alternate]);
  simPass.dispatchWorkgroups(wgCount);
  simPass.end();
  resolveTimestamp(commandEncoder, 'linkedSim');
}

export function updateDisplays(params: Record<string, number>) {
  readTimestamp('construct').then((time) => {
    params.cell = time;
  });
  readTimestamp('llValence').then((time) => {
    params.valence = time;
  });
  readTimestamp('linkedSim').then((time) => {
    params.sim = time;
  });
}
