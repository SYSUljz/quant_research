import fire

from all_dumper import DumpDataAll
from fix_dumper import DumpDataFix
from update_dumper import DumpDataUpdate

if __name__ == "__main__":
    fire.Fire(
        {
            "dump_all": DumpDataAll,
            "dump_fix": DumpDataFix,
            "dump_update": DumpDataUpdate,
        }
    )
