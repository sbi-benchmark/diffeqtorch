using ArgParse, PackageCompiler


function parse_commandline()
    s = ArgParseSettings(description="Builds system image for diffeqtorch")

    @add_arg_table! s begin
        "--replace"
            help = "Build system image, replacing default"
            action = :store_true
        "--restore"
            help = "Restore default system image"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    packages = [:DifferentialEquations, :DiffEqSensitivity]

    if parsed_args["restore"]
        println("Restore default system image")
        PackageCompiler.restore_default_sysimage()
    elseif parsed_args["replace"]
        println("Build system image, replace default")
        create_sysimage(packages, precompile_execution_file=string(@__DIR__, "/precompile.jl"), replace_default=true)
    else
        println("Build system image")
        create_sysimage(packages, sysimage_path=ENV["JULIA_SYSIMAGE_DIFFEQTORCH"], precompile_execution_file=string(@__DIR__, "/precompile.jl"))
    end

end

main()
