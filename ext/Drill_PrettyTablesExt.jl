module Drill_PrettyTablesExt

using Drill
using PrettyTables

function Drill.print_training_table(cache::Drill.RLCache)
    rows = Drill.training_metric_rows(cache)
    isempty(rows) && return nothing
    names = first.(rows)
    values = last.(rows)
    pretty_table(
        hcat(names, values);
        column_labels = ["metric", "value"],
        alignment = [:l, :r],
    )
    return nothing
end

end
