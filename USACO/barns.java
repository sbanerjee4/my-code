import java.util.*;
import java.io.*;
import java.lang.*;

public class barns
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("barns.in"));
      int t = in.nextInt();
      for(int i = 0; i < t; i++)
      {
         int n = in.nextInt(), m = in.nextInt();
         Path[] a = new Path[m];
         for(int i = 0; i < m; i++)
            a[i] = new Path(in.nextInt(), in.nextInt());
         Arrays.sort(a);
      }
      in.close();
   }
   
   static class Path
   {
      public int start;
      public int end;
      public Path(int a, int b)
      {
         start = a;
         end = b;
      }
   }
   
   static class ComparePath implements Comparator<Path>
   {
      public int compare(Path a, Path b)
      {
         return Integer.compare(a.start, b.start);
      }
   }
}