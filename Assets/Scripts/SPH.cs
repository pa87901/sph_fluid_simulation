using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

[System.Serializable]
[StructLayout(LayoutKind.Sequential, Size=44)] // So that the particle can be sent to the compute shader.
// Class for particle object definition.
// For Navier-Stokes it must be influenced by pressure,
// density, external force, have a velocity and position
// vectors.
public struct Particle {
    public float pressure; // 4
    public float density; // 8
    public Vector3 currentForce; //20
    public Vector3 velocity; // 22
    public Vector3 position; // 44 total bytes
}

public class SPH : MonoBehaviour
{
    // Attributes/variables that we want to adjust in the SPH simulator.
    [Header("General")]
    public Transform collisionSphere;
    public bool showSpheres = true;
    public Vector3Int numToSpawn = new Vector3Int(10, 10, 10);
    private int totalParticles {
        get {
            return numToSpawn.x * numToSpawn.y * numToSpawn.z;
        }
    }
    public Vector3 boxSize = new Vector3(4, 10, 3); // Size of the container for the particles to be bound within.
    public Vector3 spawnCenter; // The starting point where the particles will spawn from.
    public float particleRadius = 0.1f;
    public float spawnJitter = 0.2f;

    // These are specifically for rendering the GPU instance spheres.
    [Header("Particle Rendering")]
    public Mesh particleMesh;
    public float particleRenderSize = 8f;
    public Material material;

    [Header("Compute")]
    public ComputeShader shader;
    public Particle[] particles; // For storing the list of spawned particles to be rendered in the scene.

    // Connect Compute Shader to this SPH script.
    [Header("Fluid Constants")]
    public float boundDamping = -0.3f;
    public float viscosity = -0.003f;
    public float particleMass = 1f;
    public float gasConstant = 2f;
    public float restingDensity = 1f;
    public float timestep = 0.007f;

    // Private variables
    private ComputeBuffer _argsBuffer; // Arguments list for the GPU instance spheres.
    public ComputeBuffer _particlesBuffer; // Contains all the particles in the simulation.
    private int integrateKernel; // For integrate function to reuse it throughout our script.
    private int computeForceKernel;
    private int densityPressureKernel;

    // Set up all the variables for the Compute Shader.
    private void SetupComputeBuffers() {
        integrateKernel = shader.FindKernel("Integrate");
        computeForceKernel = shader.FindKernel("ComputeForces");
        densityPressureKernel = shader.FindKernel("ComputeDensityPressure");

        shader.SetInt("particleLength", totalParticles);
        shader.SetFloat("particleMass", particleMass);
        shader.SetFloat("viscosity", viscosity);
        shader.SetFloat("gasConstant", gasConstant);
        shader.SetFloat("restDensity", restingDensity);
        shader.SetFloat("boundDamping", boundDamping);
        shader.SetFloat("pi", Mathf.PI);
        shader.SetVector("boxSize", boxSize);

        shader.SetFloat("radius", particleRadius);
        shader.SetFloat("radius2", particleRadius * particleRadius);
        shader.SetFloat("radius3", particleRadius * particleRadius * particleRadius);
        shader.SetFloat("radius4", particleRadius * particleRadius * particleRadius * particleRadius);
        shader.SetFloat("radius5", particleRadius * particleRadius * particleRadius * particleRadius * particleRadius);

        shader.SetBuffer(integrateKernel, "_particles", _particlesBuffer); // Link the particle buffer which needs to be linked per kernel (functions that do the computation).
        shader.SetBuffer(computeForceKernel, "_particles", _particlesBuffer);
        shader.SetBuffer(densityPressureKernel, "_particles", _particlesBuffer);
    }

    // Native method.
    private void Awake() {

        SpawnParticlesInBox(); // Call the method to spawn the particles in the grid structure.

        // Set up arguments for GPU Instanced Particle Rendering.
        uint[] args = {
            particleMesh.GetIndexCount(0),
            (uint)totalParticles,
            particleMesh.GetIndexStart(0),
            particleMesh.GetBaseVertex(0),
            0
        };

        _argsBuffer = new ComputeBuffer(
            1,
            args.Length * sizeof(uint),
            ComputeBufferType.IndirectArguments
        );
        _argsBuffer.SetData(args);

        // Setup Particle (Compute) Buffer.
        _particlesBuffer = new ComputeBuffer(totalParticles, 44);
        _particlesBuffer.SetData(particles);

        SetupComputeBuffers();
    }

    private void FixedUpdate() {
        shader.SetVector("boxSize", boxSize);
        shader.SetFloat("timestep", timestep);
        shader.SetVector("spherePos", collisionSphere.transform.position);
        shader.SetFloat("sphereRadius", collisionSphere.transform.localScale.x / 2);

        // Total Particles has to be divisible by 100, because we are using 100 threads in the SPHCompute.compute shader script to ensure the function runs once per particle.
        shader.Dispatch(densityPressureKernel, totalParticles / 100, 1, 1);
        shader.Dispatch(computeForceKernel, totalParticles / 100, 1, 1);
        shader.Dispatch(integrateKernel, totalParticles / 100, 1, 1);
    }

    // Method for spawning the particles.
    private void SpawnParticlesInBox() {
        Vector3 spawnPoint  = spawnCenter;
        List<Particle> _particles = new List<Particle>();

        // Spawn particles in a grid structure.
        for (int x = 0; x < numToSpawn.x; x++) {
            for (int y = 0; y < numToSpawn.y; y++) {
                for (int z = 0; z < numToSpawn.z; z++) {
                    Vector3 spawnPos = spawnPoint + new Vector3(
                        x * particleRadius * 2,
                        y * particleRadius * 2,
                        z * particleRadius * 2
                    );
                    spawnPos += Random.onUnitSphere * particleRadius * spawnJitter;
                    Particle p  = new Particle {
                        position = spawnPos
                    };

                    _particles.Add(p); // Add them to the compute buffer array.
                }
            }
        }

        particles = _particles.ToArray();
    }

    // Native method. Define the box for the SPH simulator that will bound the spawned particles and fluid.
    private void OnDrawGizmos() {
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(Vector3.zero, boxSize);

        if (!Application.isPlaying) {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(spawnCenter, 0.1f);
        }
    }

    // Properties that link to the variables defined in the grid particle Shader included in the base project.
    private static readonly int SizeProperty = Shader.PropertyToID("_size");
    private static readonly int ParticleBufferProperty = Shader.PropertyToID("_particlesBuffer");

    // Native method. Render GPU instance particles in scene each frame.
    private void Update() {
        // Render the particles each frame to ensure the shader knows the state of the fluid.
        material.SetFloat(SizeProperty, particleRenderSize);
        // Send the list of particles to the shader, and to update the positions of particles in the compute shader.
        // We therefore do not have to extract the data from the GPU back to the CPU - V EXPENSIVE.
        material.SetBuffer(ParticleBufferProperty, _particlesBuffer);

        if (showSpheres) {
            // Unity method to render the instance meshes.
            Graphics.DrawMeshInstancedIndirect (
                particleMesh,
                0,
                material,
                new Bounds(Vector3.zero, boxSize),
                _argsBuffer,
                castShadows: UnityEngine.Rendering.ShadowCastingMode.Off
            );
        }
    }
}
